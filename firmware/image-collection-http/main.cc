/**
 * Stream images via Ethernet over USB
 *
 * Based on the following code example from Google LLC:
 *  - https://github.com/google-coral/coralmicro/tree/main/examples/camera_streaming_http
 *
 * 
 * Author: Google LLC
 * Modified by: Shawn Hymel
 * Date: November 26, 2023
 *
 * Copyright 2022 Google LLC
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

#include "libs/base/http_server.h"
#include "libs/base/led.h"
#include "libs/base/strings.h"
#include "libs/base/utils.h"
#include "libs/camera/camera.h"
#include "libs/libjpeg/jpeg.h"
#include "third_party/freertos_kernel/include/FreeRTOS.h"
#include "third_party/freertos_kernel/include/task.h"

#if defined(CAMERA_STREAMING_HTTP_ETHERNET)
#include "libs/base/ethernet.h"
#include "third_party/nxp/rt1176-sdk/middleware/lwip/src/include/lwip/prot/dhcp.h"
#elif defined(CAMERA_STREAMING_HTTP_WIFI)
#include "libs/base/wifi.h"
#endif  // defined(CAMERA_STREAMING_HTTP_ETHERNET)

// Hosts an RPC server on the Dev Board Micro that streams camera images
// to a connected client app.

namespace coralmicro {
namespace {

constexpr char kIndexFileName[] = "/index.html";
constexpr char kCameraStreamUrlPrefix[] = "/camera_stream";

// Camera settings
constexpr auto camRotation = CameraRotation::k270; // Default: CameraRotation::k0
constexpr int camWidth = 320; // Default: CameraTask::kWidth
constexpr int camHeight = 320; // Default: CameraTask::kHeight

/**
 * Handle HTTP requests
 */
HttpServer::Content UriHandler(const char* uri) {
  if (StrEndsWith(uri, "index.shtml") ||
      StrEndsWith(uri, "coral_micro_camera.html")) {
    return std::string(kIndexFileName);
  } else if (StrEndsWith(uri, kCameraStreamUrlPrefix)) {

    // Turn status LED on to let the user know we're taking a photo
    LedSet(Led::kUser, true);

    // Buffer to hold image
    std::vector<uint8_t> buf(camWidth * camHeight *
                             CameraFormatBpp(CameraFormat::kRgb));

    // Configure camera image
    auto fmt = CameraFrameFormat{
        CameraFormat::kRgb,       
        CameraFilterMethod::kBilinear,
        camRotation,       
        camWidth,
        camHeight,
        /*preserve_ratio=*/false, 
        buf.data(),
        /*while_balance=*/true};

    // Take a photo
    if (!CameraTask::GetSingleton()->GetFrame({fmt})) {
      printf("Unable to get frame from camera\r\n");
      return {};
    }

    // Compress image to JPEG
    std::vector<uint8_t> jpeg;
    JpegCompressRgb(buf.data(), fmt.width, fmt.height, /*quality=*/75, &jpeg);

    // Turn status LED off to let the user know we're done taking a photo
    LedSet(Led::kUser, false);

    return jpeg;
  }
  return {};
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
  printf("Image collection HTTP server over USB\r\n");

  // Initialize camera
  CameraTask::GetSingleton()->SetPower(true);
  CameraTask::GetSingleton()->Enable(CameraMode::kStreaming);

  // Initialize IP over USB
  std::string usb_ip;
  if (GetUsbIpAddress(&usb_ip)) {
    printf("Serving on: http://%s\r\n", usb_ip.c_str());
  }

  // Initialize HTTP server (attach request handler)
  HttpServer http_server;
  http_server.AddUriHandler(UriHandler);
  UseHttpServer(&http_server);

  vTaskSuspend(nullptr);
}
}  // namespace
}  // namespace coralmicro

extern "C" void app_main(void* param) {
  (void)param;
  coralmicro::Main();
}

