//
// Created by sean on 7/17/19.
//

#ifndef RESDET_DETECTOR_H
#define RESDET_DETECTOR_H

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

class Detector {
private:
    unsigned int num_classes;
    int top_k;
    float nms_thresh;
    unsigned int tub;
    int ssd_dim;
    torch::Tensor output;
    std::vector<cv::Scalar> color{cv::Scalar(255,0,0), cv::Scalar(0, 255,255), cv::Scalar(0,0,255), cv::Scalar(255,0,255)};
public:
    Detector(unsigned int, int, float, unsigned int, int);
    void log_params();
//    torch::Tensor detect(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
//            std::vector<float>, float, bool);
    torch::Tensor detect(const torch::Tensor&, const torch::Tensor&, std::vector<float>);
    std::tuple<torch::Tensor, int> nms(const torch::Tensor&, const torch::Tensor&);
    void visualization(cv::Mat, const torch::Tensor&);
    ~Detector(){};
};


#endif //RESDET_DETECTOR_H
