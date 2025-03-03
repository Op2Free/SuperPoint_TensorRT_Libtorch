#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "logger.h"

#include "SuperPointNet.h"
#include "TorchFrontend.h"
#include "Drawer.h"

int main(int argc, const char *argv[])
{
    std::string logLevelStr = getLogLevelFromEnvironment();
    spdlog::level::level_enum logLevel = toSpdlogLevel(logLevelStr);
    spdlog::set_level(logLevel);

    if(argc != 4)
    {
        std::cerr << std::endl << "Usage: ./match_torch path_to_torchModel[.pt] height width" << std::endl;
        return 1;
    }
    
    std::string torch_weight = std::string(argv[1]);

    float nn_thresh = 0.7;
    //use SuperPointNet or torch::jit::script::Module
    TorchFrontend<torch::jit::script::Module> spfrontend(torch_weight, 4, 0.015, nn_thresh);
    
    // Read the input image
    // TODO: You will need to read the input image required for your model
    const std::string img_path1 = "../data/imageA.png";
    const std::string img_path2 = "../data/imageB.png";
    auto img1 = cv::imread(img_path1);
    auto img2 = cv::imread(img_path2);

    if (img1.empty()) {
        const std::string msg = "Unable to read image at path: " + img_path1;
        spdlog::error(msg);
        throw std::runtime_error(msg);
    }
    if (img2.empty()) {
        const std::string msg = "Unable to read image at path: " + img_path2;
        spdlog::error(msg);
        throw std::runtime_error(msg);
    }

    cv::resize(img1, img1, {std::atoi(argv[3]), std::atoi(argv[2])});
    cv::resize(img2, img2, {std::atoi(argv[3]), std::atoi(argv[2])});
    
    cv::Mat gray1, gray2;
    cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2, gray2, cv::COLOR_BGR2GRAY);
    gray1.convertTo(gray1, CV_32F, 1.0f/255.0f);
    gray2.convertTo(gray2, CV_32F, 1.0f/255.0f);

    torch::Tensor pts1, pts2;
    torch::Tensor desc1, desc2;
    spfrontend.run(gray1, pts1, desc1);
    spfrontend.run(gray2, pts2, desc2);
    
    cv::Mat cimg1 = img1.clone();
    cv::Mat cimg2 = img2.clone();
    drawKpts(cimg1, pts1);
    drawKpts(cimg2, pts2);
    cv::imshow("imageA", cimg1);
    cv::imshow("imageb", cimg2);

    torch::Tensor matches;
    matches = nn_match_two_way(desc1, desc2, nn_thresh);

    torch::Tensor mkpts1, mkpts2;
    mkpts1 = pts1.index({matches.index({0, "..."}).to(torch::kInt64), torch::indexing::Slice(0,2)});
    mkpts2 = pts2.index({matches.index({1, "..."}).to(torch::kInt64), torch::indexing::Slice(0,2)});


    cv::Mat matching = drawMatch(cimg1, cimg2, mkpts1, mkpts2);
    cv::imshow("Matching", matching);
    cv::waitKey(0);

    return 0;
}
