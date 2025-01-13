#ifndef SUPERPOINT_NET_H
#define SUPERPOINT_NET_H

#include <torch/torch.h>

class SuperPointNet : public torch::nn::Module {
public:
    SuperPointNet();

    std::vector<torch::Tensor> forward(torch::Tensor x);

private:
    const int c1 = 64, c2 = 64, c3 = 128, c4 = 128, c5 = 256, d1 = 256;

    torch::nn::ReLU relu;
    torch::nn::MaxPool2d pool;
    
    torch::nn::Conv2d conv1a;
    torch::nn::Conv2d conv1b;
    torch::nn::Conv2d conv2a;
    torch::nn::Conv2d conv2b;
    torch::nn::Conv2d conv3a;
    torch::nn::Conv2d conv3b;
    torch::nn::Conv2d conv4a;
    torch::nn::Conv2d conv4b;

    torch::nn::Conv2d convPa;
    torch::nn::Conv2d convPb;

    torch::nn::Conv2d convDa;
    torch::nn::Conv2d convDb;

};
#endif
