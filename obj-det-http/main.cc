// #include "libs/base/http_server.h"

// #include "libs/base/led.h"
// #include "libs/base/strings.h"
// #include "libs/base/tasks.h"
// #include "libs/base/utils.h"
// #include "third_party/freertos_kernel/include/FreeRTOS.h"
// #include "third_party/freertos_kernel/include/task.h"

// namespace coralmicro {
// namespace {

// HttpServer::Content UriHandler(const char* path) {
//   printf("Request received for %s\r\n", path);
//   std::vector<uint8_t> html;
//   html.reserve(64);
//   if (std::strcmp(path, "/hello.html") == 0) {
//     printf("Hello World!\r\n");
//     StrAppend(&html, "<html><body>Hello World!</body></html>");
//     return html;
//   }
//   return {};
// }

// [[noreturn]] void blink_task(void* param) {
//   auto led_type = static_cast<Led*>(param);
//   printf("Hello task.\r\n");
//   bool on = true;
//   while (true) {
//     on = !on;
//     LedSet(*led_type, on);
//     vTaskDelay(pdMS_TO_TICKS(500));
//   }
// }

// void Main() {
//   printf("HTTP Server Example!\r\n");
//   // Turn on Status LED to show the board is on.
//   LedSet(Led::kStatus, true);

//   auto user_led = Led::kUser;
//   xTaskCreate(&blink_task, "blink_user_led_task", configMINIMAL_STACK_SIZE,
//               &user_led, kAppTaskPriority, nullptr);

//   printf("Starting server...\r\n");
//   HttpServer http_server;
//   http_server.AddUriHandler(UriHandler);
//   UseHttpServer(&http_server);

//   std::string ip;
//   if (GetUsbIpAddress(&ip)) {
//     printf("GO TO:   http://%s/%s\r\n", ip.c_str(), "hello.html");
//   }
//   vTaskSuspend(nullptr);
// }

// }  // namespace
// }  // namespace coralmicro

// extern "C" void app_main(void* param) {
//   (void)param;
//   coralmicro::Main();
//   vTaskSuspend(nullptr);void Blink(unsigned int num, unsigned int delay_ms) {






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
* Author: Shawn Hymel
* Date: November 12, 2023
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

namespace coralmicro {
namespace {

// Globals
constexpr char kIndexFileName[] = "/coral_micro_camera.html";
constexpr char kCameraStreamUrlPrefix[] = "/camera_stream";
constexpr char kModelPath[] =
    "/models/tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite";
constexpr int kTensorArenaSize = 8 * 1024 * 1024;
STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, kTensorArenaSize);
static std::vector<uint8_t> *img_ptr;
static SemaphoreHandle_t img_mutex;
static unsigned int img_width;
static unsigned int img_height;

/*******************************************************************************
* Functions
*/

void Blink(unsigned int num, unsigned int delay_ms);

// Handle HTTP requests
HttpServer::Content UriHandler(const char* uri) {

  // Give client main page
  if (StrEndsWith(uri, "index.shtml") ||
      StrEndsWith(uri, "coral_micro_camera.html")) {
    return std::string(kIndexFileName);

  // Give client compressed image data
  } else if (StrEndsWith(uri, kCameraStreamUrlPrefix)) {
    
    // TODO: we're sending some weird interlaced RGB/BGR crap here. What's going on?
    // Read image from shared memory and compress to JPG
    std::vector<uint8_t> jpeg;
    if (xSemaphoreTake(img_mutex, portMAX_DELAY) == pdTRUE) {
      JpegCompressRgb(
        img_ptr->data(), 
        img_width, 
        img_height, 
        75,         // Quality
        &jpeg
      );
      xSemaphoreGive(img_mutex);
    }

    return jpeg;
  }
  return {};
}

/**
* Capture image and perform inference
*/
bool DetectFromCamera(
  tflite::MicroInterpreter* interpreter, 
  int model_width,
  int model_height,
  std::vector<tensorflow::Object>* results,
  std::vector<uint8>* image
) {

  // Make sure memory has been allocated for results and image
  CHECK(results != nullptr);
  CHECK(image != nullptr);

  // Turn status LED on to let the user know we're taking a photo
  LedSet(Led::kUser, true);

  // Get pointer to input tensor buffer
  auto* input_tensor = interpreter->input_tensor(0);

  // Configure image from camera
  // printf("a: %lu\r\n", xTaskGetTickCount() * (1000 / configTICK_RATE_HZ));
  CameraFrameFormat fmt{
    CameraFormat::kRgb,   
    CameraFilterMethod::kBilinear,
    CameraRotation::k270, 
    model_width,
    model_height,         
    false,
    image->data()
  };

  // Get frame from camera using the configuration we set (~38 ms)
  if (xSemaphoreTake(img_mutex, portMAX_DELAY) == pdTRUE) {
    if (!CameraTask::GetSingleton()->GetFrame({fmt})) {
      return false;
    }
    xSemaphoreGive(img_mutex);
  }

  // Turn status LED off to let the user know we're done taking a photo
  LedSet(Led::kUser, false);

  // Copy image to input tensor (~6 ms)
  std::memcpy(
    tflite::GetTensorData<uint8_t>(input_tensor), 
    image->data(),
    image->size()
  );

  // Perform inference (~65 ms)
  if (interpreter->Invoke() != kTfLiteOk) {
    return false;
  }
  
  // Get object detection results (as a vector of Objects)
  *results = tensorflow::GetDetectionResults(interpreter, 0.6, 5);

  return true;
}

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

  // Do forever
  while (true) {
    
    // Calculate time between inferences
    timestamp = xTaskGetTickCount() * (1000 / configTICK_RATE_HZ);
    dtime = timestamp - timestamp_prev;
    timestamp_prev = timestamp;

    // Get image and perform inference
    printf("Taking photo...\r\n");
    if (DetectFromCamera(&interpreter, 
      img_width, 
      img_height, 
      &results, 
      img_ptr)
    ) {
      std::string output = "bboxes | dtime: " + std::to_string(dtime) + "\r\n";
      for (const auto& object : results) {
        output += "  id: " + std::to_string(object.id) +
          " | score: " + std::to_string(object.score) +
          " | xmin: " + std::to_string(object.bbox.xmin) +
          " | ymin: " + std::to_string(object.bbox.ymin) +
          " | xmax: " + std::to_string(object.bbox.xmax) +
          " | ymin: " + std::to_string(object.bbox.ymax) + 
          "\r\n";
      }
      printf("%s", output.c_str());

    } else {
      printf("Failed to detect image from camera.\r\n");
    }
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
  printf("Object detection inference and HTTP server over USB\r\n");

  // Initialize mutex
  img_mutex = xSemaphoreCreateMutex();
  if (img_mutex == NULL) {
    printf("Error creating mutex\r\n");
    while (true) {
      Blink(2, 100);
    }
  }

  // Initialize camera
  CameraTask::GetSingleton()->SetPower(true);
  CameraTask::GetSingleton()->Enable(CameraMode::kStreaming);

  // Start capture and inference task
  printf("Starting inference task\r\n");
  xTaskCreate(
    &InferenceTask,
    "InferenceTask",
    configMINIMAL_STACK_SIZE * 30,
    nullptr,
    kAppTaskPriority,
    nullptr
  );

  // Initialize IP over USB
  std::string usb_ip;
  if (GetUsbIpAddress(&usb_ip)) {
    printf("Serving on: http://%s\r\n", usb_ip.c_str());
  }

  // Initialize HTTP server (attach request handler)
  HttpServer http_server;
  http_server.AddUriHandler(UriHandler);
  UseHttpServer(&http_server);

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
