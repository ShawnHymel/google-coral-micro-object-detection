/**
 * Object Detection with HTTP server over USB
 *
 * Perform object detection with the provided TFLite file running on the TPU. 
 * Image and object detection data is streamed to an HTTP server that can be 
 * accessed via Ethernet-over-USB connection.
 *
 * Based on the following code examples from Google LLC:
 *  - https://github.com/google-coral/coralmicro/tree/main/examples/camera_streaming_http
 *  - https://github.com/google-coral/coralmicro/tree/main/examples/detect_objects
 *
 * WARNING: There is a bug in HTTP handler part of this code that seems to 
 * corrupt the bounding box info when requested by the client. I suspect this is
 * due to the server not being able to handle multiple requests at the same time
 * or there is a race condition with the bounding box info being updated. If you
 * solve this, please let me know. You will get a metaphorical gold star.
 * 
 * Author: Shawn Hymel
 * Date: November 25, 2023
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstdio>
#include <vector>

#include "libs/base/filesystem.h"
#include "libs/base/http_server.h"
#include "libs/base/led.h"
#include "libs/base/strings.h"
#include "libs/base/utils.h"
#include "libs/camera/camera.h"
#include "libs/libjpeg/jpeg.h"
#include "libs/tensorflow/detection.h"
#include "libs/tensorflow/utils.h"
#include "libs/tpu/edgetpu_manager.h"
#include "libs/tpu/edgetpu_op.h"
#include "third_party/freertos_kernel/include/FreeRTOS.h"
#include "third_party/freertos_kernel/include/task.h"
#include "third_party/freertos_kernel/include/semphr.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_error_reporter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_interpreter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_mutable_op_resolver.h"

#define ENABLE_HTTP_SERVER 1
#define DEBUG 1

namespace coralmicro {
namespace {

// Struct
typedef struct {
  std::string info;
  std::vector<uint8_t> *jpeg;
} ImgResult;

// Camera settings
constexpr auto camRotation = CameraRotation::k270; // Default: CameraRotation::k0

// Globals
constexpr char kIndexFileName[] = "/index.html";
constexpr char kCameraStreamUrlPrefix[] = "/camera_stream";
constexpr char kBoundingBoxPrefix[] = "/bboxes";
constexpr char kModelPath[] =
    "/model_int8_edgetpu.tflite";
constexpr int kTensorArenaSize = 8 * 1024 * 1024;
STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, kTensorArenaSize);
static std::vector<uint8_t> *img_ptr;
static SemaphoreHandle_t img_mutex;
static SemaphoreHandle_t bbox_mutex;
static int img_width;
static int img_height;
static constexpr float bbox_threshold = 0.2;
static constexpr int max_bboxes = 5;
static constexpr unsigned int bbox_buf_size = 100 + (max_bboxes * 200) + 1;
static char bbox_buf[bbox_buf_size];

// Copy of image data for HTTP server
#if ENABLE_HTTP_SERVER
static std::vector<uint8_t> *img_copy;
#endif

/*******************************************************************************
 * Functions
 */

void Blink(unsigned int num, unsigned int delay_ms);

#if ENABLE_HTTP_SERVER
/**
 * Handle HTTP requests
 */
HttpServer::Content UriHandler(const char* uri) {

  // Give client main page
  if (StrEndsWith(uri, "index.shtml") ||
      StrEndsWith(uri, "coral_micro_camera.html")) {
    return std::string(kIndexFileName);

  // Give client compressed image data
  } else if (StrEndsWith(uri, kCameraStreamUrlPrefix)) {
    
    // Read image from shared memory and compress to JPG
    std::vector<uint8_t> jpeg;
    if (xSemaphoreTake(img_mutex, portMAX_DELAY) == pdTRUE) {
      JpegCompressRgb(
        img_copy->data(), 
        img_width, 
        img_height, 
        75,         // Quality
        &jpeg
      );
      xSemaphoreGive(img_mutex);
    }

    return jpeg;

  // Give client bounding box info
  } else if (StrEndsWith(uri, kBoundingBoxPrefix)) {

    // Read bounding box info from shared memory and convert to vector of bytes
    char bbox_info_copy[bbox_buf_size];
    std::vector<uint8_t> bbox_info_bytes;
    if (xSemaphoreTake(bbox_mutex, portMAX_DELAY) == pdTRUE) {
      std::strcpy(bbox_info_copy, bbox_buf);
      bbox_info_bytes.assign(
        bbox_info_copy, 
        bbox_info_copy + std::strlen(bbox_info_copy)
      );
      xSemaphoreGive(bbox_mutex);
    }

    // TODO: Figure out the multi-request or race condition bug that is causing
    // the bbox_info_bytes to be corrupted. The workaround is to have the
    // client timeout if it doesn't get a response in some amount of time.

    return bbox_info_bytes;
  }

  return {};
}
#endif

/**
 * Loop forever taking images from the camera and performing inference
 */
[[noreturn]] void InferenceTask(void* param) {

  // Used for calculating FPS
  unsigned long dtime;
  unsigned long timestamp;
  unsigned long timestamp_prev = xTaskGetTickCount() * 
    (1000 / configTICK_RATE_HZ);

  // Load model
  std::vector<uint8_t> model;
  if (!LfsReadFile(kModelPath, &model)) {
    printf("ERROR: Failed to load %s\r\n", kModelPath);
    vTaskSuspend(nullptr);
  }

  // Initialize TPU
  auto tpu_context = EdgeTpuManager::GetSingleton()->OpenDevice();
  if (!tpu_context) {
    printf("ERROR: Failed to get EdgeTpu context\r\n");
    vTaskSuspend(nullptr);
  }

  // Initialize ops
  tflite::MicroErrorReporter error_reporter;
  tflite::MicroMutableOpResolver<3> resolver;
  resolver.AddDequantize();
  resolver.AddDetectionPostprocess();
  resolver.AddCustom(kCustomOp, RegisterCustomOp());

  // Initialize TFLM interpreter for inference
  tflite::MicroInterpreter interpreter(
    tflite::GetModel(model.data()), 
    resolver,
    tensor_arena, 
    kTensorArenaSize,
    &error_reporter
  );
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    printf("ERROR: AllocateTensors() failed\r\n");
    vTaskSuspend(nullptr);
  }

  // Check model input tensor size
  if (interpreter.inputs().size() != 1) {
    printf("ERROR: Model must have only one input tensor\r\n");
    vTaskSuspend(nullptr);
  }

  // Configure model inputs and outputs
  auto* input_tensor = interpreter.input_tensor(0);
  img_height = input_tensor->dims->data[1];
  img_width = input_tensor->dims->data[2];
  img_ptr = new std::vector<uint8>(img_height * img_width * 
    CameraFormatBpp(CameraFormat::kRgb));
  std::vector<tensorflow::Object> results;
#if ENABLE_HTTP_SERVER
    img_copy = new std::vector<uint8_t>(img_ptr->size());
#endif

// #if DEBUG
//   printf("TensorFlow Lite version: %s\r\n", tflite::GetVersion());
// #endif

  // Get output tensor shapes
  TfLiteTensor* tensor_bboxes = interpreter.output_tensor(0);
  TfLiteTensor* tensor_scores = interpreter.output_tensor(1);
  unsigned int num_boxes = tensor_bboxes->dims->data[1];
  unsigned int num_coords = tensor_bboxes->dims->data[2];
  unsigned int num_classes = tensor_scores->dims->data[2];

  // Print output tensor shapes
#if DEBUG
  printf("num_boxes: %d\r\n", num_boxes);
  printf("num_coords: %d\r\n", num_coords);
  printf("num_classes: %d\r\n", num_classes);
  printf("bytes in tensor_bboxes: %d\r\n", tensor_bboxes->bytes);
  if (tensor_scores->data.data == nullptr) {
    printf("tensor_scores.data is empty!\r\n");
  }
#endif

  // Convert threshold to fixed point
  uint8_t threshold = static_cast<uint8_t>(bbox_threshold * 256);

  // Do forever
  while (true) {

    // Calculate time between inferences
    timestamp = xTaskGetTickCount() * (1000 / configTICK_RATE_HZ);
    dtime = timestamp - timestamp_prev;
    timestamp_prev = timestamp;

    // Turn status LED on to let the user know we're taking a photo
    LedSet(Led::kUser, true);

    // Get frame from camera using the configuration we set (~38 ms)
    if (xSemaphoreTake(img_mutex, portMAX_DELAY) == pdTRUE) {

      // Configure camera image
      CameraFrameFormat fmt{
        CameraFormat::kRgb,   
        CameraFilterMethod::kBilinear,
        camRotation,
        img_height,
        img_width,         
        false,            // Preserve ratio
        img_ptr->data(),  // Where the image is saved
        true              // Auto white balance
      };

      // Take a photo
      if (!CameraTask::GetSingleton()->GetFrame({fmt})) {
        printf("ERROR: Could not capture frame from camera\r\n");
        continue;
      }

      // Turn status LED off to let the user know we're done taking a photo
      LedSet(Led::kUser, false);

      // Copy image to input tensor (~6 ms)
      std::memcpy(
        tflite::GetTensorData<uint8_t>(input_tensor), 
        img_ptr->data(),
        img_ptr->size()
      );

      // Perform inference (~65 ms)
      if (interpreter.Invoke() != kTfLiteOk) {
        printf("ERROR: Inference failed\r\n");
        continue;
      }
    
      // Get object detection results (as a vector of Objects)
      // results = tensorflow::GetDetectionResults(&interpreter, bbox_threshold, max_bboxes);


      // ***
      // TODO: Figure out why this is hanging on ->data.uint8

      // for (unsigned int i = 0; i < tensor_bboxes->dims->size; i++) {
      //   printf("tensor_bboxes->dims->data[%d]: %d\r\n", i, tensor_bboxes->dims->data[i]);
      // }

      // if (tensor_scores->type == kTfLiteUInt8) {
      //   printf("tensor_scores.type == kTfLiteUInt8\r\n");
      // } else {
      //   printf("tensor_scores.type != kTfLiteUInt8\r\n");
      // }

      // printf("bytes in tensor_bboxes: %d\r\n", tensor_bboxes->bytes);

      // if (tensor_scores->data.data == nullptr) {
      //   printf("tensor_scores.data is empty!\r\n");
      // }

      // BBox data
      float score, class_max, ymin, xmin, ymax, xmax;
      std::vector<std::vector<float>> bbox_list;

      // Get data
      uint8_t *scores = tensor_scores->data.uint8;
      uint8_t *bboxes = tensor_bboxes->data.uint8;
      
      // Find bounding boxes with scores above threshold
      for (unsigned int i = 0; i < num_boxes; ++i) {
        score = -1.0f;
        class_max = 0.0;
        for (unsigned int j = 0; j < num_classes; ++j) {

          // Compare score to threshold and find max score
          if (scores[i * 3 + j] >= threshold) {
            if (scores[i * 3 + j] > score) {
              score = (float)scores[i * 3 + j] * 0.00390625;
              class_max = (float)j;
            }
          }
        }

        // If score is above threshold, add to list of bboxes
        if (score > 0.0f) {
          ymin = (float)bboxes[i * 4] * 0.00390625;
          xmin = (float)bboxes[i * 4 + 1] * 0.00390625;
          ymax = (float)bboxes[i * 4 + 2] * 0.00390625;
          xmax = (float)bboxes[i * 4 + 3] * 0.00390625;
          bbox_list.push_back({score, class_max, ymin, xmin, ymax, xmax});
        }
      }

      // Sort bboxes by score
      std::sort(bbox_list.begin(), bbox_list.end(), 
        [](const std::vector<float>& a, const std::vector<float>& b) {
          return a[0] > b[0];
        }
      );

      // Print bboxes
      for (const auto& bbox : bbox_list) {
        printf("score: %f, class: %f, ymin: %f, xmin: %f, ymax: %f, xmax: %f\r\n", 
          bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]);
      }

      // TODO: 
      //  - get bounding box info from score indices
      //  - push score, class, bbox info to results vector
      //  - sort results vector by score
      //  - return top X results
      //  - convert results to JSON string


      // printf("scores: ");
      // for (unsigned int i = 0; i < 19125; ++i) {
      //   printf("|");
      //   for (unsigned int j = 0; j < 3; ++i) {
      //     printf("%f ", scores[i * 3 + j]);
      //   }
      // }
      // printf("\r\n");


      // TEST: Get results from inference
      // const float *bboxes, *scores;
      // bboxes = tflite::GetTensorData<float>(interpreter.output_tensor(0));
      // scores = tflite::GetTensorData<float>(interpreter.output_tensor(1));
      
      // printf("bbox scores: ");
      // for (unsigned int i = 0; i < 19125; ++i) {
      //   for (unsigned int j = 0; j < 3; ++j) {
      //     printf("%f ", scores[i * 3 + j]);
      //   }
      //   printf("\r\n");
      // }
      // printf("\r\n");

      // ***
      
      // Copy image to separate buffer for HTTP server
#if ENABLE_HTTP_SERVER
      std::memcpy(
        img_copy->data(),
        img_ptr->data(),
        img_ptr->size()
      );
#endif

      // Unlock critical section
      xSemaphoreGive(img_mutex);
    }

    // TEST
    printf("dtime: %lu\r\n", dtime);

    // // Convert results into json string
    // std::string bbox_string = "{\"dtime\": " + std::to_string(dtime) + ", ";
    //   bbox_string += "\"bboxes\": [";
    //   for (const auto& object : results) {
    //     bbox_string += "{\"id\": " + std::to_string(object.id) + ", ";
    //     bbox_string += "\"score\": " + std::to_string(object.score) + ", ";
    //     bbox_string += "\"xmin\": " + std::to_string(object.bbox.xmin) + ", ";
    //     bbox_string += "\"ymin\": " + std::to_string(object.bbox.ymin) + ", ";
    //     bbox_string += "\"xmax\": " + std::to_string(object.bbox.xmax) + ", ";
    //     bbox_string += "\"ymax\": " + std::to_string(object.bbox.ymax) + "}";
    //     if (&object != &results.back()) {
    //       bbox_string += ", ";
    //     }
    //   }
    //   bbox_string += "]}";

    // // Check length of JSON string
    // if (bbox_string.length() > bbox_buf_size) {
    //   printf("ERROR: Bounding box JSON string too long\r\n");
    //   continue;
    // }

    // // Convert global char array
    // if (xSemaphoreTake(bbox_mutex, portMAX_DELAY) == pdTRUE) {
    //   std::strcpy(bbox_buf, bbox_string.c_str());
    //   xSemaphoreGive(bbox_mutex);
    // }

    // // Print bounding box JSON
    // printf("%s\r\n", bbox_buf);

    // Sleep to let other tasks run
    // vTaskDelay(pdMS_TO_TICKS(10));
  }
}

/**
* Blink error codes on the status LED
*/
void Blink(unsigned int num, unsigned int delay_ms) {
  static bool on = false;
  for (unsigned int i = 0; i < num * 2; i++) {
    on = !on;
    coralmicro::LedSet(Led::kStatus, on);
    vTaskDelay(pdMS_TO_TICKS(delay_ms));
  }
}

/*******************************************************************************
 * Main
 */

void Main() {

  // Say hello
  Blink(3, 500);
#if DEBUG
  printf("Object detection inference and HTTP server over USB\r\n");
#endif

  // Initialize image mutex
  img_mutex = xSemaphoreCreateMutex();
  if (img_mutex == NULL) {
    printf("Error creating image mutex\r\n");
    while (true) {
      Blink(2, 100);
    }
  }

  // Initialize bounding box mutex
  bbox_mutex = xSemaphoreCreateMutex();
  if (bbox_mutex == NULL) {
    printf("Error creating bbox mutex\r\n");
    while (true) {
      Blink(2, 100);
    }
  }

  // Initialize camera
  CameraTask::GetSingleton()->SetPower(true);
  CameraTask::GetSingleton()->Enable(CameraMode::kStreaming);

  // Start capture and inference task
#if DEBUG
  printf("Starting inference task\r\n");
#endif
  xTaskCreate(
    &InferenceTask,
    "InferenceTask",
    configMINIMAL_STACK_SIZE * 30,
    nullptr,
    kAppTaskPriority - 1, // Console and server are same priority as app, so make inference lower
    nullptr
  );

#if ENABLE_HTTP_SERVER

  // Initialize IP over USB
  std::string usb_ip;
  if (GetUsbIpAddress(&usb_ip)) {
#if DEBUG
    printf("Serving on: http://%s\r\n", usb_ip.c_str());
#endif
  }

  // Initialize HTTP server (attach request handler)
  HttpServer http_server;
  http_server.AddUriHandler(UriHandler);
  UseHttpServer(&http_server);

#endif

  // Main will go to sleep
  vTaskSuspend(nullptr);
}

}  // namespace
}  // namespace coralmicro

/**
 * Entrypoint
 */
extern "C" void app_main(void* param) {
  (void)param;
  coralmicro::Main();
}
