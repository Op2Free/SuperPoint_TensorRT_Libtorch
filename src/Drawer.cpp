#include "Drawer.h"
torch::Tensor nn_match_two_way(const torch::Tensor &desc1, const torch::Tensor &desc2, float nn_thresh){
    if((desc1.size(0) == 0) || (desc2.size(0) == 0)){
        return torch::zeros({3, 0});
    }
    // Compute L2 distance. Easy since vectors are unit normalized.
    torch::Tensor dmat = desc1.mm(desc2.transpose(0,1));
    dmat = (2-2*(dmat.clip(-1, 1))).sqrt().to(torch::kCPU);
    // Get NN indices and scores.
    torch::Tensor idx = dmat.argmin(1, false); 
    torch::Tensor scores = dmat.index({torch::arange(dmat.size(0)), idx});
    // Threshold the NN matches.
    torch::Tensor keep = scores < nn_thresh;
    // Check if nearest neighbor goes both directions and keep those.
    torch::Tensor idx2 = dmat.argmin(0, false);
    torch::Tensor keep_bi = (torch::arange(idx.size(0)) == idx2.index({idx}));
    keep = keep.logical_and(keep_bi);
    idx = idx.index({keep});
    scores = scores.index({keep});
    // Get the surviving point indices.
    torch::Tensor m_idx1 = torch::arange(desc1.size(0)).index({keep});
    torch::Tensor m_idx2 = idx;
    // Populate the final 3xN match data structure.
    torch::Tensor matches = torch::zeros({3, keep.sum().item<int>()});
    matches[0] = m_idx1;
    matches[1] = m_idx2;
    matches[2] = scores;
    return matches;
}

void drawKpts(cv::Mat &img, const torch::Tensor &kpts){
    for(int i=0; i<kpts.size(0); ++i){
        cv::Point pt(kpts[i][0].item<double>(), kpts[i][1].item<double>());
        cv::circle(img, pt, 2, CV_RGB(0, 0, 255), -1);
    }
}

cv::Mat drawMatch(cv::Mat &img1, cv::Mat &img2, const torch::Tensor &mkpts1, const torch::Tensor &mkpts2){
    cv::Mat pair(img1.rows, img1.cols + img2.cols, img1.type());

    // 将 image1 的数据复制到 combinedImage
    img1.copyTo(pair(cv::Rect(0, 0, img1.cols, img1.rows)));
    img2.copyTo(pair(cv::Rect(img1.cols, 0, img2.cols, img2.rows)));
    
    for(int i=0; i<mkpts1.size(0); ++i){
        cv::Point mkpt1(mkpts1[i][0].item<double>(), mkpts1[i][1].item<double>());
        cv::Point mkpt2((mkpts2[i][0].item<double>() + (double)img1.cols), mkpts2[i][1].item<double>());
        // cv::circle(pair, mkpt1, 3, CV_RGB(0, 0, 255), 1);
        // cv::circle(pair, mkpt2, 3, CV_RGB(0, 0, 255), 1);
        cv::line(pair, mkpt1, mkpt2, CV_RGB(0, 255, 0), 2);
    }
    return pair;
}

