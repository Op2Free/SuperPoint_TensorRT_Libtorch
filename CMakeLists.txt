cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(superpoint_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -DNDEBUG -Wno-deprecated-declarations")

# TODO: Specify the path to TensorRT root dir
set(TensorRT_DIR "/opt/TensorRT-10.7.0.23/")
# Build the TensorRT inference engine library
# TensorRT is found and linked by the tensorrt-cpp-api
add_subdirectory(libs/tensorrt-cpp-api)

# TODO: Specify the path to libtorch root dir
set(Torch_DIR "/opt/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

find_package(OpenCV REQUIRED)

# Customize the tensorrt-cpp-api to grayscale input
add_library(custom_trt_api SHARED libs/customEngine.cpp)
target_link_libraries(custom_trt_api PUBLIC tensorrt_cpp_api)
target_include_directories(custom_trt_api PUBLIC libs/tensorrt-cpp-api/src)

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
)

add_library(superpoint_cpp SHARED
    src/TRTFrontend.cpp
    src/Drawer.cpp
    src/SuperPointNet.cpp
    src/TorchFrontend.cpp
)
target_link_libraries(superpoint_cpp
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
    custom_trt_api
)
target_include_directories(superpoint_cpp PUBLIC ${PROJECT_SOURCE_DIR}/libs)


# Benchmark TensorRT and Libtorch
add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark superpoint_cpp)

# Matching by TensorRT
add_executable(match_trt match_trt.cpp)
target_link_libraries(match_trt superpoint_cpp)

# Matching by Libtorch
add_executable(match_torch match_torch.cpp)
target_link_libraries(match_torch superpoint_cpp)



