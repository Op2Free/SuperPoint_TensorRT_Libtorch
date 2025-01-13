#ifndef TRTFRONTEND_H
#define TRTFRONTEND_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <vector>
#include <algorithm>
#include <string>

#include "customEngine.h"
#include "logger.h"

#include <torch/torch.h>


bool ends_with(const std::string& str, const std::string& suffix);

class TRTFrontend
{
public:
    TRTFrontend(const std::string &modelPath, float nms_dist_, float conf_thresh_, float nn_thresh_);

    torch::Tensor nms_fast(const torch::Tensor &pts, int H, int W, float dist_thresh);
    
    void run(const cv::Mat& cpuImg, torch::Tensor& pts, torch::Tensor& desc);

private:
    std::shared_ptr<customEngine<float>> engine;
    float nms_dist;
    float conf_thresh;
    float nn_thresh;

    int cell = 8;
    int border_remove = 4;

    torch::Device device = torch::kCPU;
};

#endif