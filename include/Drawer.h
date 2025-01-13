#ifndef DRAWER_H
#define DRAWER_H

#include<torch/torch.h>
#include<opencv2/opencv.hpp>

torch::Tensor nn_match_two_way(const torch::Tensor &desc1, const torch::Tensor &desc2, float nn_thresh);

void drawKpts(cv::Mat &img, const torch::Tensor &kpts);

cv::Mat drawMatch(cv::Mat &img1, cv::Mat &img2, const torch::Tensor &mkpts1, const torch::Tensor &mkpts2);

#endif