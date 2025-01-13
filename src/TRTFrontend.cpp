#include "TRTFrontend.h"

bool ends_with(const std::string& str, const std::string& suffix) {
    if (str.size() < suffix.size()) {
        return false;
    }
    return str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

TRTFrontend::TRTFrontend(const std::string &modelPath, float nms_dist_, float conf_thresh_, float nn_thresh_)
    : nms_dist(nms_dist_), conf_thresh(conf_thresh_), nn_thresh(nn_thresh_)
{
    if(torch::cuda::is_available()) device = torch::kCUDA;
    
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

    engine = std::make_shared<customEngine<float>>(options);
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

    // following params:
    //    subVals = {0.5f, 0.5f, 0.5f};
    //    divVals = {0.5f, 0.5f, 0.5f};
    //    normalize = true;

    if (ends_with(modelPath, ".onnx")) {
        // Build the onnx model into a TensorRT engine file, and load the TensorRT
        // engine file into memory.
        bool succ = engine->buildLoadNetwork(modelPath, subVals, divVals, normalize);
        if (!succ) {
            throw std::runtime_error("Unable to build or load TensorRT engine.");
        }
    } else if (ends_with(modelPath, ".engine")) {
        // Load the TensorRT engine file directly
        bool succ = engine->loadNetwork(modelPath, subVals, divVals, normalize);
        if (!succ) {
            const std::string msg = "Unable to load TensorRT engine.";
            spdlog::error(msg);
            throw std::runtime_error(msg);
        }
    }
}


void TRTFrontend::run(const cv::Mat& cpuImg, torch::Tensor& pts, torch::Tensor& desc)
{
    // Upload the image GPU memory
    cv::cuda::GpuMat img;
    img.upload(cpuImg);

    // The model expects gray input
    cv::cuda::cvtColor(img, img, cv::COLOR_BGR2GRAY);

    // In the following section we populate the input vectors to later pass for
    // inference
    const auto &inputDims = engine->getInputDims();
    std::vector<std::vector<cv::cuda::GpuMat>> inputs;

    // Let's use a batch size which matches that which we set the
    // Options.optBatchSize option
    // size_t batchSize = options.optBatchSize;

    // TODO:
    // For the sake of the demo, we will be feeding the same image to all the
    // inputs You should populate your inputs appropriately.
    for (const auto &inputDim : inputDims) { // For each of the model inputs...
        std::vector<cv::cuda::GpuMat> input;
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
        cv::cuda::resize(img, resized, cv::Size(inputDim.d[2],inputDim.d[1]));
        input.emplace_back(std::move(resized));
        
        inputs.emplace_back(std::move(input));
    }

    std::vector<std::vector<std::vector<float>>> featureVectors; //[batch, outputs, ...]

    bool succ = engine->runInference(inputs, featureVectors);
    if (!succ) {
        const std::string msg = "Unable to run inference.";
        spdlog::error(msg);
        throw std::runtime_error(msg);
    }

    int H = cpuImg.rows;
    int W = cpuImg.cols;

    const auto &outputDims = engine->getOutputDims();

    torch::Tensor semi, coarse_desc;
    semi = torch::from_blob(featureVectors[0][0].data(), {1, outputDims[0].d[1], outputDims[0].d[2], outputDims[0].d[3]}, torch::kFloat32);
    coarse_desc = torch::from_blob(featureVectors[0][1].data(), {1, outputDims[1].d[1], outputDims[1].d[2], outputDims[1].d[3]}, torch::kFloat32);
   
    // --- Process points.  
    semi = semi.squeeze();  // [65,Hc,Wc]
    torch::Tensor dense = torch::nn::functional::softmax(semi, torch::nn::functional::SoftmaxFuncOptions(0));

    // Remove dustbin.
    torch::Tensor nodust = dense.slice(0, 0, 64); // [64,Hc,Wc]

    // Reshape to get full resolution heatmap.
    int Hc = int(H / cell);
    int Wc = int(W / cell);
    nodust = nodust.permute({1, 2, 0}); // [H,W,64]

    torch::Tensor heatmap = nodust.reshape({Hc, Wc, cell, cell});
    heatmap = heatmap.permute({0, 2, 1, 3});
    heatmap = heatmap.reshape({Hc * cell, Wc * cell}); // [H,W]

    auto yx = (heatmap >= conf_thresh).nonzero();
    if(yx.size(0)==0){
        return;
    }

    auto xs = yx.index({"...",1}).clone();
    auto ys = yx.index({"...",0}).clone();

    pts = torch::zeros({3, xs.size(0)});
    pts[0] = xs;
    pts[1] = ys;
    pts[2] = heatmap.index({ys, xs}).clone();

    pts = nms_fast(pts, H, W, nms_dist); // Apply NMS. (Already sorted by confidence)
    // Remove points along border.
    int bord = border_remove;
    auto toremoveW = (pts[0] < bord).logical_or(pts[0] >= (W-bord));
    auto toremoveH = (pts[1] < bord).logical_or(pts[1] >= (H-bord));
    auto toremove = toremoveW.logical_or(toremoveH);
    pts = pts.index({"...", toremove.logical_not()}).transpose(0,1);
    // --- Process descriptor.
    int D = coarse_desc.size(1);
    if(pts.size(0) == 0)
    {
        desc = torch::zeros({D, 0}, torch::kFloat64);
        return;
    }
    else
    {   
        coarse_desc = coarse_desc.to(device);

        // Interpolate into descriptor map using 2D point locations.
        torch::Tensor samp_pts = pts.index({"...", torch::indexing::Slice(0,2)}).clone();
        samp_pts.index({"...", 0}) = (samp_pts.index({"...", 0}) / ((float)W/2)) - 1;
        samp_pts.index({"...", 1}) = (samp_pts.index({"...", 1}) / ((float)H/2)) - 1;
        samp_pts = samp_pts.view({1, 1, -1, 2}).to(torch::kFloat).to(device);
        
        torch::nn::functional::GridSampleFuncOptions options;
        options.mode(torch::kBilinear);
        options.padding_mode(torch::kZeros);
        options.align_corners(true);

        desc = torch::nn::functional::grid_sample(coarse_desc, samp_pts, options);
        desc = desc.reshape({D, -1}).transpose(0,1);
        desc = torch::nn::functional::normalize(desc, 
                                                torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
    }
}

torch::Tensor TRTFrontend::nms_fast(const torch::Tensor &in_corners, int H, int W, float dist_thresh)
{
    // Track NMS data.
    torch::Tensor grid = torch::zeros({H, W}, torch::kInt32);
    torch::Tensor inds = torch::zeros({H, W}, torch::kInt64);
    // Sort by confidence and round to nearest int.
    torch::Tensor inds1 = in_corners[2].argsort(0, true);
    torch::Tensor corners = in_corners.index({"...", inds1}).clone();
    torch::Tensor rcorners = corners.index({torch::indexing::Slice(torch::indexing::None,2), "..."})
                                    .clone().round().to(torch::kInt32); //Rounded corners.
    // Check for edge case of 0 or 1 corners.
    if(rcorners.size(1) == 0){
        return torch::zeros({3,0}, torch::kInt32);
    }
    if(rcorners.size(1) == 1){
        return torch::cat({rcorners, in_corners[2].unsqueeze(0)},0).reshape({3,1});
    } 

    // Initialize the grid.
    for(int i=0; i<rcorners.size(1); ++i){
        grid[rcorners[1][i]][rcorners[0][i]] = 1;
        inds[rcorners[1][i]][rcorners[0][i]] = i;
    }
    // Pad the border of the grid, so that we can NMS points near the border.
    int pad = dist_thresh;
    torch::nn::ConstantPad2d padder(torch::nn::ConstantPad2dOptions(pad, 0));
    grid = padder(grid);
    // Iterate through points, highest to lowest conf, suppress neighborhood.
    for(int i=0; i<rcorners.size(1); ++i){
        // Account for top and left padding.
        int pt[2] = {rcorners[0][i].item<int>()+pad, rcorners[1][i].item<int>()+pad};
        if(grid[pt[1]][pt[0]].item<int>() == 1){ // If not yet suppressed.
            grid.index_put_({torch::indexing::Slice(pt[1]-pad, pt[1]+pad+1), 
                            torch::indexing::Slice(pt[0]-pad, pt[0]+pad+1)}, 0);
            grid[pt[1]][pt[0]] = -1;
        }
    }
    // Get all surviving -1's and return sorted array of remaining corners.
    torch::Tensor keepyx = (grid==-1).nonzero();
    torch::Tensor keepy = keepyx.index({"...",0}) - pad;
    torch::Tensor keepx = keepyx.index({"...",1}) - pad;
    torch::Tensor inds_keep = inds.index({keepy, keepx});
    torch::Tensor out = corners.index({"...", inds_keep.to(torch::kInt64)});
    torch::Tensor inds2 = out[2].argsort(0, true); // Sort by confidence
    out = out.index({"...", inds2});
    return out;
}

