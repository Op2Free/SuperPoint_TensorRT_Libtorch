#include "logger.h"

#include <chrono>

#include "TRTFrontend.h"
#include "TorchFrontend.h"


int main(int argc, char *argv[]) {

    if(argc != 3)
    {
        std::cerr << std::endl << "Usage: ./benchmark path_to_trtModel[.onnx or .engine] path_to_torchModel" << std::endl;
        return 1;
    }
    
    std::string trt_weight = std::string(argv[1]);
    std::string torch_weight = std::string(argv[2]);

    // Read the input image
    const std::string img_path = "../data/imageA.png";
    auto img = cv::imread(img_path);
    if (img.empty()) {
        const std::string msg = "Unable to read image at path: " + img_path;
        spdlog::error(msg);
        throw std::runtime_error(msg);
    }

    if (!Util::doesFileExist(trt_weight)) {
        auto msg = "Error, unable to read model for TensorRT at path: " + trt_weight;
        spdlog::error(msg);
        throw std::runtime_error(msg);
    }

    if (!Util::doesFileExist(torch_weight)) {
        auto msg = "Error, unable to read model for Libtorch at path: " + torch_weight;
        spdlog::error(msg);
        throw std::runtime_error(msg);
    }

    std::string logLevelStr = getLogLevelFromEnvironment();
    spdlog::level::level_enum logLevel = toSpdlogLevel(logLevelStr);
    spdlog::set_level(logLevel);

    // Specify our GPU inference configuration options
    Options options;
    // Specify what precision to use for inference
    // FP16 is approximately twice as fast as FP32.
    options.precision = Precision::FP32;
    // If using INT8 precision, must specify path to directory containing
    // calibration data.
    options.calibrationDataDirectoryPath = "";
    // Specify the batch size to optimize for.
    options.optBatchSize = 1;
    // Specify the maximum batch size we plan on running.
    options.maxBatchSize = 1;
    // Specify the directory where you want the model engine model file saved.
    options.engineFileDir = ".";

    customEngine<float> engine(options);

    // Define our preprocessing code
    // The default Engine::build method will normalize values between [0.f, 1.f]
    // Setting the normalize flag to false will leave values between [0.f, 255.f]
    // (some converted models may require this).

    // For our YoloV8 model, we need the values to be normalized between
    // [0.f, 1.f] so we use the following params
    // std::array<float, 3> subVals{0.f, 0.f, 0.f};
    // std::array<float, 3> divVals{1.f, 1.f, 1.f};

    // For grayscale input like SuperPoint
    std::array<float, 1> subVals{0.f};
    std::array<float, 1> divVals{1.f};
    bool normalize = true;
    // Note, we could have also used the default values.

    // If the model requires values to be normalized between [-1.f, 1.f], use the
    // following params:
    //    subVals = {0.5f, 0.5f, 0.5f};
    //    divVals = {0.5f, 0.5f, 0.5f};
    //    normalize = true;

    if (ends_with(trt_weight, ".onnx")) {
        // Build the onnx model into a TensorRT engine file, and load the TensorRT
        // engine file into memory.
        bool succ = engine.buildLoadNetwork(trt_weight, subVals, divVals, normalize);
        if (!succ) {
            throw std::runtime_error("Unable to build or load TensorRT engine.");
        }
    } else if (ends_with(trt_weight, ".engine")) {
        // Load the TensorRT engine file directly
        bool succ = engine.loadNetwork(trt_weight, subVals, divVals, normalize);
        if (!succ) {
            const std::string msg = "Unable to load TensorRT engine.";
            spdlog::error(msg);
            throw std::runtime_error(msg);
        }
    }


    spdlog::info("Benchmarking TensorRT.");

    // Upload the image GPU memory
    cv::cuda::GpuMat gpuImg;
    gpuImg.upload(img);

    // The model expects gray input
    cv::cuda::cvtColor(gpuImg, gpuImg, cv::COLOR_BGR2GRAY);

    // In the following section we populate the input vectors to later pass for
    // inference
    const auto &inputDims = engine.getInputDims();
    std::vector<std::vector<cv::cuda::GpuMat>> inputs;

    // Let's use a batch size which matches that which we set the
    // Options.optBatchSize option
    size_t batchSize = options.optBatchSize;

    // TODO:
    // For the sake of the demo, we will be feeding the same image to all the
    // inputs You should populate your inputs appropriately.
    for (const auto &inputDim : inputDims) { // For each of the model inputs...
        std::vector<cv::cuda::GpuMat> input;
        for (size_t j = 0; j < batchSize; ++j) { // For each element we want to add to the batch...
            // TODO:
            // You can choose to resize by scaling, adding padding, or a combination
            // of the two in order to maintain the aspect ratio You can use the
            // Engine::resizeKeepAspectRatioPadRightBottom to resize to a square while
            // maintain the aspect ratio (adds padding where necessary to achieve
            // this).
            // auto resized = Engine<float>::resizeKeepAspectRatioPadRightBottom(img, inputDim.d[1], inputDim.d[2]);
            // You could also perform a resize operation without maintaining aspect
            // ratio with the use of padding by using the following instead:
            //            cv::cuda::resize(img, resized, cv::Size(inputDim.d[2],
            //            inputDim.d[1])); // TRT dims are (height, width) whereas
            //            OpenCV is (width, height)
            cv::cuda::GpuMat resized;
            cv::cuda::resize(gpuImg, resized, cv::Size(inputDim.d[2],inputDim.d[1]));
            input.emplace_back(std::move(resized));
        }
        inputs.emplace_back(std::move(input));
    }

    // Warm up the network before we begin the benchmark
    spdlog::info("Warming up the network...");
    std::vector<std::vector<std::vector<float>>> featureVectors;
    for (int i = 0; i < 100; ++i) {
        bool succ = engine.runInference(inputs, featureVectors);
        if (!succ) {
            const std::string msg = "Unable to run inference.";
            spdlog::error(msg);
            throw std::runtime_error(msg);
        }
    }

    // Benchmark the inference time
    size_t numIterations = 1000;
    spdlog::info("Running benchmarks ({} iterations)...\n", numIterations);
    preciseStopwatch stopwatch;
    for (size_t i = 0; i < numIterations; ++i) {
        featureVectors.clear();
        engine.runInference(inputs, featureVectors);
    }
    auto totalElapsedTimeMs = stopwatch.elapsedTime<float, std::chrono::milliseconds>();
    auto avgElapsedTimeMs = totalElapsedTimeMs / numIterations / static_cast<float>(inputs[0].size());

    spdlog::info("====Benchmarking TensorRT complete====");
    spdlog::info("Avg time per sample: ");
    spdlog::info("Avg time per sample: {} ms", avgElapsedTimeMs);
    spdlog::info("Batch size: {}", inputs[0].size());
    spdlog::info("Avg FPS: {} fps", static_cast<int>(1000 / avgElapsedTimeMs));
    spdlog::info("======================================\n");

    spdlog::info("Benchmarking Libtorch.");

    torch::Device device = torch::kCPU;
    if(torch::cuda::is_available()){
        device = torch::kCUDA;
    }
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    gray.convertTo(gray, CV_32F, 1.0f/255.0f);

    int H = gray.rows;
    int W = gray.cols;
    // auto img_ = img.clone();
    auto inp = torch::from_blob(gray.data, {1, 1, H, W}, torch::kFloat32).to(device);
    
    //use SuperPointNet or torch::jit::script::Module
    TorchFrontend<torch::jit::script::Module> torchfrontend(torch_weight, 4, 0.015, 0.7);

    torch::Tensor semi, coarse_desc;

    // Warm up the network before we begin the benchmark
    spdlog::info("Warming up the network...");
    for (int i = 0; i < 100; ++i) {
        torchfrontend.modelForward(inp, semi, coarse_desc);
    }

    // Benchmark the inference time
    spdlog::info("Running benchmarks ({} iterations)...\n", numIterations);
    preciseStopwatch stopwatch_torch;
    for (size_t i = 0; i < numIterations; ++i) {
        torchfrontend.modelForward(inp, semi, coarse_desc);
    }
    totalElapsedTimeMs = stopwatch_torch.elapsedTime<float, std::chrono::milliseconds>();
    avgElapsedTimeMs = totalElapsedTimeMs / numIterations;

    spdlog::info("Benchmarking complete!");
    spdlog::info("====Benchmarking Libtorch complete====");
    spdlog::info("Avg time per sample: ");
    spdlog::info("Avg time per sample: {} ms", avgElapsedTimeMs);
    spdlog::info("Avg FPS: {} fps", static_cast<int>(1000 / avgElapsedTimeMs));
    spdlog::info("======================================\n");

    return 0;
}