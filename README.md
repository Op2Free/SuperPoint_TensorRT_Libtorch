# SuperPoint_TensorRT_Libtorch
This project is a C++ version of [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork). It covers the following:

- Step by step preparation from [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork) to C++ implementation.
- Benchmarking the implementations between Pytorch, Libtorch, TensorRT.
- Matching with Libtorch.
- Matching with TensorRT.

<img src="data/matching.png" width="520">

## Getting Started
This project is well tested on Ubuntu20.04, CUDA 11.7, cudnn 8. 

`git clone --recursive https://github.com/Op2Free/SuperPoint_TensorRT_Libtorch.git`

### Prerequisites
- This project makes use of [tensorrt-cpp-api](https://github.com/cyrusbehr/tensorrt-cpp-api.git) to run TensorRT inference. It has been included in `libs` . Following the prerequisites in [tensorrt-cpp-api](https://github.com/cyrusbehr/tensorrt-cpp-api.git), suitable CUDA, cudnn, OpenCV, TensorRT should be ready.
- [Libtorch](https://pytorch.org/cppdocs/installing.html). Version 1.13 is tested. Installing usually involves unzipping the file only.
- Python environment: Pytorch (1.13 tested), [ONNX](https://onnxruntime.ai/docs/install/) (1.16.2 tested)， opencv-python
- Modify `TensorRT_DIR` and `Torch_DIR` in `CMakeLists.txt` with the path to your TensorRT and Libtorch.

### Exporting weight
Libtorch loads models with [TorchScript](https://pytorch.org/tutorials/advanced/cpp_export.html). TensorRT takes in models exported from ONNX.

- `cd scripts`
- `python exportJIT.py` It will load the original `.pth` file and save as `.pt` in `weights`.
- `python exportONNX.py 480 640` Specify the height and width of the images to be used. In our example pair (imageA.png, imageB.png), we use 640x480. `.onnx` is saved in `weights`.
- (optinal) TensorRT can also load `.engine`. Using [trtexec](https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html#convert-onnx-engine) with only one command line. Otherwise, TensorRT uses API to load `.onnx` and does the conversion.

### Building the project
- `mkdir build`
- `cd build`
- `cmake ..`
- `make -j`

Since [tensorrt-cpp-api](https://github.com/cyrusbehr/tensorrt-cpp-api.git) only provides api for RGB images, I customize it to grayscale images. To understand the code in `libs`, it's better to understand [tensorrt-cpp-api](https://github.com/cyrusbehr/tensorrt-cpp-api.git) first.

### Running the Executable
- `cd build`
- Benchmarking Libtorch and TensorRT `./benchmark ../weights/superpoint.onnx ../weights/superpoint.pt`
- Matching with TensorRT `./match_trt ../weights/superpoint.onnx 480 640`
- Matching with Libtorch `./match_torch ../weights/superpoint.pt 480 640`. Use `SuperPointNet` (define your own forward function) or `torch::jit::script::Module` (direct convertion) in `TorchFrontend<T>`.

### Benchmarks
Benchmarks run on GeForceMX450, without taking post processing into accout.

| Implementation   | Precision | Avg Inference Time |
|---------|-----------|--------------------|
| Pytorch | FP32      |  37.403 ms           |
| Libtorch | FP32      |  38.045 ms           |
| TensorRT | FP32      |  25.552 ms           |

### Frequent issue
```
[2025-01-08 12:01:18.830] [error] createInferBuilder: Error Code 6: API Usage Error (Unable to load library: libnvinfer_builder_resource.so.10.7.0: libnvinfer_builder_resource.so.10.7.0: cannot open shared object file: No such file or directory)
```

To solve this, `export LD_LIBRARY_PATH=<TensorRT-${version}/lib>:$LD_LIBRARY_PATH`
