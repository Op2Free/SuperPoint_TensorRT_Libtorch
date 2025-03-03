#include "logger.h"

#include <opencv2/opencv.hpp>

#include "TRTFrontend.h"
#include "Drawer.h"

int main(int argc, char *argv[]) {

    std::string logLevelStr = getLogLevelFromEnvironment();
    spdlog::level::level_enum logLevel = toSpdlogLevel(logLevelStr);
    spdlog::set_level(logLevel);

    if(argc != 4)
    {
        std::cerr << std::endl << "Usage: ./match_trt path_to_trtModel[.onnx or .engine] height width" << std::endl;
        return 1;
    }
    
    std::string trt_weight = std::string(argv[1]);

    float nn_thresh = 0.7;
    TRTFrontend spfrontend(trt_weight, 4, 0.015, nn_thresh);
    
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

    torch::Tensor pts1, pts2;
    torch::Tensor desc1, desc2;
    spfrontend.run(img1, pts1, desc1);
    spfrontend.run(img2, pts2, desc2);
    
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
