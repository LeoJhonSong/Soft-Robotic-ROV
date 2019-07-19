//
// Created by sean on 7/16/19.
//

#ifndef RESDET_UTILS_H
#define RESDET_UTILS_H

#endif //RESDET_UTILS_H

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

cv::Mat tensor2im(torch::Tensor tensor);