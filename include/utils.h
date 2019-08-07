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
#include "ruas.h"

cv::Mat tensor2im(torch::Tensor tensor);
void clip(float& n, float lower, float upper);
void clip(int& n, int lower, int upper);
void parse_key(int, bool&, bool&, std::vector<float>&, int&, int&, CFilt&);
void raw_write();
