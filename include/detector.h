//
// Created by sean on 7/17/19.
//

#ifndef RESDET_DETECTOR_H
#define RESDET_DETECTOR_H

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <utils.h>

class Detector {
private:
    unsigned int num_classes;
    int top_k;
    float nms_thresh;
    unsigned int tub;
    int ssd_dim;
    torch::Tensor output;
    std::vector<std::map<int, std::pair<torch::Tensor, int>>> tubelets;
    std::vector<std::vector<int>> ides;
    torch::Tensor history_max_ides;
    unsigned int hold_len;
//    unsigned feature_size;
    std::vector<cv::Scalar> color{cv::Scalar(255,0,0), cv::Scalar(0, 255,255), cv::Scalar(0,0,255), cv::Scalar(255,0,255)};
public:
    Detector(unsigned int, int, float, unsigned char, int);
    void log_params();
//    torch::Tensor detect(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
//            std::vector<float>, float, bool);
    torch::Tensor detect(const torch::Tensor&, const torch::Tensor&, std::vector<float>);
    torch::Tensor detect(const torch::Tensor&, const torch::Tensor&, std::vector<float>, float, bool);
    std::tuple<torch::Tensor, int> nms(const torch::Tensor&, const torch::Tensor&);
    torch::Tensor iou(const torch::Tensor&, unsigned char);
    void visualization(cv::Mat&, const torch::Tensor&);
    void init_tubelets();
    void delete_tubelets(unsigned char);
    ~Detector(){};
};


#endif //RESDET_DETECTOR_H
