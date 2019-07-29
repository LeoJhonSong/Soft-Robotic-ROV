//
// Created by sean on 7/11/19.
//

#include <utils.h>


cv::Mat tensor2im(torch::Tensor tensor, std::vector<int> vis_size) {
    tensor = torch::upsample_bilinear2d(tensor, {vis_size.at(1), vis_size.at(0)}, true);
//    tensor = tensor[0].add(128.0).permute({1,2,0}).to(torch::kU8).to(torch::kCPU);
    tensor = tensor[0].add(1.0).div(2.0).mul(255.0).permute({1,2,0}).to(torch::kU8).to(torch::kCPU);
    cv::Mat img(tensor.size(0), tensor.size(1), CV_8UC3);
    std::memcpy((void*)img.data, tensor.data_ptr(), sizeof(torch::kU8)*tensor.numel());
    cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
//    cv::resize(img, img, vis_size);
    return img;
}

int clip(int n, int lower, int upper) {
    return std::max(lower, std::min(n, upper));
}