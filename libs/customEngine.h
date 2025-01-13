#pragma once

#include "engine.h"
#include "customInt8Calibrator.h"
#undef CHECK


template <typename T>
class customEngine : public IEngine<T> {
public:
    customEngine(const Options &options);
    ~customEngine();

    // Build the onnx model into a TensorRT engine file, cache the model to disk
    // (to avoid rebuilding in future), and then load the model into memory The
    // default implementation will normalize values between [0.f, 1.f] Setting the
    // normalize flag to false will leave values between [0.f, 255.f] (some
    // converted models may require this). If the model requires values to be
    // normalized between [-1.f, 1.f], use the following params:
    //    subVals = {0.5f, 0.5f, 0.5f};
    //    divVals = {0.5f, 0.5f, 0.5f};
    //    normalize = true;
    virtual bool buildLoadNetwork(std::string onnxModelPath, const std::array<float, 3> &subVals = {0.f, 0.f, 0.f},
                                  const std::array<float, 3> &divVals = {1.f, 1.f, 1.f}, bool normalize = true){
                                    throw std::logic_error("Function should not be called.");
                                  }
    virtual bool loadNetwork(std::string trtModelPath, const std::array<float, 3> &subVals = {0.f, 0.f, 0.f},
                             const std::array<float, 3> &divVals = {1.f, 1.f, 1.f}, bool normalize = true){
                                throw std::logic_error("Function should not be called.");
                             }

    bool buildLoadNetwork(std::string onnxModelPath, const std::array<float, 1> &subVals = {0.f},
                          const std::array<float, 1> &divVals = {1.f}, bool normalize = true);

    // Load a TensorRT engine file from disk into memory
    // The default implementation will normalize values between [0.f, 1.f]
    // Setting the normalize flag to false will leave values between [0.f, 255.f]
    // (some converted models may require this). If the model requires values to
    // be normalized between [-1.f, 1.f], use the following params:
    //    subVals = {0.5f, 0.5f, 0.5f};
    //    divVals = {0.5f, 0.5f, 0.5f};
    //    normalize = true;
    bool loadNetwork(std::string trtModelPath, const std::array<float, 1> &subVals = {0.f},
                     const std::array<float, 1> &divVals = {1.f}, bool normalize = true);

    // Run inference.
    // Input format [input][batch][cv::cuda::GpuMat]
    // Output format [batch][output][feature_vector]
    bool runInference(const std::vector<std::vector<cv::cuda::GpuMat>> &inputs, std::vector<std::vector<std::vector<T>>> &featureVectors) override;

    // Utility method for resizing an image while maintaining the aspect ratio by
    // adding padding to smaller dimension after scaling While letterbox padding
    // normally adds padding to top & bottom, or left & right sides, this
    // implementation only adds padding to the right or bottom side This is done
    // so that it's easier to convert detected coordinates (ex. YOLO model) back
    // to the original reference frame.
    static cv::cuda::GpuMat resizeKeepAspectRatioPadRightBottom(const cv::cuda::GpuMat &input, size_t height, size_t width,
                                                                const cv::Scalar &bgcolor = cv::Scalar(0, 0, 0));

    [[nodiscard]] const std::vector<nvinfer1::Dims3> &getInputDims() const override { return m_inputDims; };
    [[nodiscard]] const std::vector<nvinfer1::Dims> &getOutputDims() const override { return m_outputDims; };

    // Utility method for transforming triple nested output array into 2D array
    // Should be used when the output batch size is 1, but there are multiple
    // output feature vectors
    static void transformOutput(std::vector<std::vector<std::vector<T>>> &input, std::vector<std::vector<T>> &output);

    // Utility method for transforming triple nested output array into single
    // array Should be used when the output batch size is 1, and there is only a
    // single output feature vector
    static void transformOutput(std::vector<std::vector<std::vector<T>>> &input, std::vector<T> &output);
    // Convert NHWC to NCHW and apply scaling and mean subtraction
    static cv::cuda::GpuMat blobFromGpuMats(const std::vector<cv::cuda::GpuMat> &batchInput, const std::array<float, 1> &subVals,
                                            const std::array<float, 1> &divVals, bool normalize, bool swapRB = false);

private:
    // Build the network
    bool build(std::string onnxModelPath, const std::array<float, 1> &subVals, const std::array<float, 1> &divVals, bool normalize);

    // Converts the engine options into a string
    std::string serializeEngineOptions(const Options &options, const std::string &onnxModelPath);

    void getDeviceNames(std::vector<std::string> &deviceNames);

    void clearGpuBuffers();

    // Normalization, scaling, and mean subtraction of inputs
    std::array<float, 1> m_subVals{};
    std::array<float, 1> m_divVals{};
    bool m_normalize;

    // Holds pointers to the input and output GPU buffers
    std::vector<void *> m_buffers;
    std::vector<uint32_t> m_outputLengths{};
    std::vector<nvinfer1::Dims3> m_inputDims;
    std::vector<nvinfer1::Dims> m_outputDims;
    std::vector<std::string> m_IOTensorNames;
    int32_t m_inputBatchSize;

    // Must keep IRuntime around for inference, see:
    // https://forums.developer.nvidia.com/t/is-it-safe-to-deallocate-nvinfer1-iruntime-after-creating-an-nvinfer1-icudaengine-but-before-running-inference-with-said-icudaengine/255381/2?u=cyruspk4w6
    std::unique_ptr<nvinfer1::IRuntime> m_runtime = nullptr;
    std::unique_ptr<customInt8EntropyCalibrator2> m_calibrator = nullptr;
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine = nullptr;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context = nullptr;
    const Options m_options;
    Logger m_logger;
};

template <typename T> customEngine<T>::customEngine(const Options &options) : m_options(options) {}

template <typename T> customEngine<T>::~customEngine() { clearGpuBuffers(); }

// Include inline implementations
#include "customEngineRunInference.inl"
#include "customEngineUtilities.inl"
#include "customEngineBuildLoadNetwork.inl"