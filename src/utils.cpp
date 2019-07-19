//
// Created by sean on 7/11/19.
//

#include <utils.h>


cv::Mat tensor2im(torch::Tensor tensor) {
    tensor = tensor[0].add(1.0).div(2.0).mul(255.0).permute({1,2,0}).to(torch::kU8).to(torch::kCPU);
    cv::Mat img(tensor.size(0), tensor.size(1), CV_8UC3);
    std::memcpy((void*)img.data, tensor.data_ptr(), sizeof(torch::kU8)*tensor.numel());
//    std::memcpy(tensor.data_ptr(), img.data, sizeof(float)*tensor.numel());
    cv::cvtColor(img, img, cv::COLOR_RGB2BGR);

    return img;
}