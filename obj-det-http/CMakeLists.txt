cmake_minimum_required(VERSION 3.13)

# Toolchain must be set before project() call.
if (NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(
        CMAKE_TOOLCHAIN_FILE 
        $ENV{CORAL_MICRO_PATH}/cmake/toolchain-arm-none-eabi-gcc.cmake
    )
endif()

# Name the app
project(coralmicro-app)

# Set language requirements
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add Coral Micro SDK
include_directories(coralmicro)
add_subdirectory(coralmicro)

# Add the source code
add_executable_m7(
    coralmicro-app
    main.cc
    DATA
    ${PROJECT_SOURCE_DIR}/models/tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite
    ${CMAKE_CURRENT_SOURCE_DIR}/web/coral_micro_camera.html>/coral_micro_camera.html
)

# Link to libraries
target_link_libraries(
    coralmicro-app
    libs_base-m7_freertos
    libs_base-m7_http_server
    libs_jpeg_m7
)
