# include "SuperPointNet.h"

SuperPointNet::SuperPointNet()
    :   relu(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true))),
        pool(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))),

        //Encoder
        conv1a(torch::nn::Conv2d(torch::nn::Conv2dOptions(1, c1, 3).stride(1).padding(1))),
        conv1b(torch::nn::Conv2d(torch::nn::Conv2dOptions(c1, c1, 3).stride(1).padding(1))),
        conv2a(torch::nn::Conv2d(torch::nn::Conv2dOptions(c1, c2, 3).stride(1).padding(1))),
        conv2b(torch::nn::Conv2d(torch::nn::Conv2dOptions(c2, c2, 3).stride(1).padding(1))),
        conv3a(torch::nn::Conv2d(torch::nn::Conv2dOptions(c2, c3, 3).stride(1).padding(1))),
        conv3b(torch::nn::Conv2d(torch::nn::Conv2dOptions(c3, c3, 3).stride(1).padding(1))),
        conv4a(torch::nn::Conv2d(torch::nn::Conv2dOptions(c3, c4, 3).stride(1).padding(1))),
        conv4b(torch::nn::Conv2d(torch::nn::Conv2dOptions(c4, c4, 3).stride(1).padding(1))),

        //Detector
        convPa(torch::nn::Conv2d(torch::nn::Conv2dOptions(c4, c5, 3).stride(1).padding(1))),
        convPb(torch::nn::Conv2d(torch::nn::Conv2dOptions(c5, 65, 1).stride(1).padding(0))),

        //Descriptor
        convDa(torch::nn::Conv2d(torch::nn::Conv2dOptions(c4, c5, 3).stride(1).padding(1))),
        convDb(torch::nn::Conv2d(torch::nn::Conv2dOptions(c5, d1, 1).stride(1).padding(0)))
{
    register_module("relu", relu);
    register_module("pool", pool);
    register_module("conv1a", conv1a);
    register_module("conv1b", conv1b);
    register_module("conv2a", conv2a);
    register_module("conv2b", conv2b);
    register_module("conv3a", conv3a);
    register_module("conv3b", conv3b);
    register_module("conv4a", conv4a);
    register_module("conv4b", conv4b);
    register_module("convPa", convPa);
    register_module("convPb", convPb);
    register_module("convDa", convDa);
    register_module("convDb", convDb);
}

std::vector<torch::Tensor> SuperPointNet::forward(torch::Tensor x)
{

    //Encoder
    x = relu(conv1a->forward(x));
    x = relu(conv1b->forward(x));
    x = pool->forward(x);
    x = relu(conv2a->forward(x));
    x = relu(conv2b->forward(x));
    x = pool->forward(x);
    x = relu(conv3a->forward(x));
    x = relu(conv3b->forward(x));
    x = pool->forward(x);
    x = relu(conv4a->forward(x));
    x = relu(conv4b->forward(x));

    //Detector
    torch::Tensor cPa = relu(convPa->forward(x));
    torch::Tensor semi = convPb->forward(cPa); // [N, 65, H/8, W/8]

    //Descriptor
    torch::Tensor cDa = relu(convDa->forward(x));
    torch::Tensor desc = convDb->forward(cDa); // [N, 256, H/8, W/8]

    auto dn = torch::norm(desc, 2, 1);
    desc = desc.div(torch::unsqueeze(dn, 1));

    std::vector<torch::Tensor> ret;
    ret.push_back(semi);
    ret.push_back(desc);

    return ret;
}
