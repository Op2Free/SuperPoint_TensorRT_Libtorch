#ifndef TORCHFRONTEND_H
#define TORCHFRONTEND_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <vector>
#include <algorithm>

#include <torch/torch.h>
#include <torch/script.h>

#include "SuperPointNet.h" 


template<typename T>
class TorchFrontend
{
public:
    TorchFrontend(std::string weight_path, float nms_dist_, float conf_thresh_, float nn_thresh_);

    torch::Tensor nms_fast(const torch::Tensor &pts, int H, int W, float dist_thresh);
    
    void run(const cv::Mat& img, torch::Tensor& pts, torch::Tensor& desc);

    void loadModel(std::string weight_path){std::cerr<<"Not a specialised type T."<<std::endl;}
    
    void modelForward(const torch::Tensor &inp, torch::Tensor &semi, torch::Tensor &coarse_desc){std::cerr<<"Not a specialised type T."<<std::endl;}

private:
    std::shared_ptr<T> model;
    float nms_dist;
    float conf_thresh;
    float nn_thresh;

    int cell = 8;
    int border_remove = 4;
    torch::Device device = torch::kCPU;
};


template<typename T>
TorchFrontend<T>::TorchFrontend(std::string weight_path, float nms_dist_, float conf_thresh_, float nn_thresh_)
    : nms_dist(nms_dist_), conf_thresh(conf_thresh_), nn_thresh(nn_thresh_)
{
    if(torch::cuda::is_available()){
        device = torch::kCUDA;
    }
    loadModel(weight_path);
    model->to(device);
    model->eval();
}

template<>
void TorchFrontend<torch::jit::script::Module>::loadModel(std::string weight_path){
    std::cout<<"\n Using torch script."<<std::endl;
    model = std::make_shared<torch::jit::script::Module>(torch::jit::load(weight_path));
}

template<>
void TorchFrontend<SuperPointNet>::loadModel(std::string weight_path){
    std::cout<<"\n Using custom model with loaded weight."<<std::endl;
    model = std::make_shared<SuperPointNet>();
    torch::load(model, weight_path);
}

template<>
void TorchFrontend<torch::jit::script::Module>::modelForward(const torch::Tensor &inp, torch::Tensor &semi, torch::Tensor &coarse_desc){
    auto outs = model->forward({inp}).toTuple();
    semi = outs->elements()[0].toTensor();
    coarse_desc = outs->elements()[1].toTensor();
}

template<>
void TorchFrontend<SuperPointNet>::modelForward(const torch::Tensor &inp, torch::Tensor &semi, torch::Tensor &coarse_desc){
    auto outs = model->forward({inp});
    semi = outs[0];
    coarse_desc = outs[1];
}

template<typename T>
void TorchFrontend<T>::run(const cv::Mat& img, torch::Tensor& pts, torch::Tensor& desc)
{
    int H = img.rows;
    int W = img.cols;
    // auto img_ = img.clone();
    auto inp = torch::from_blob(img.data, {1, 1, H, W}, torch::kFloat32).to(device);
    
    torch::Tensor semi, coarse_desc;
    // Forward pass of network.
    modelForward(inp, semi, coarse_desc);
    // --- Process points.
    semi.to(torch::kCPU);    
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

template<typename T>
torch::Tensor TorchFrontend<T>::nms_fast(const torch::Tensor &in_corners, int H, int W, float dist_thresh)
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

#endif