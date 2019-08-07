//
// Created by sean on 7/17/19.
//

#ifndef RESDET_DETECTOR_H
#define RESDET_DETECTOR_H

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <utils.h>
#include <uart.h>
#include <sstream>
#include <iomanip>


class Detector {
private:
    unsigned int num_classes;
    int top_k;
    float nms_thresh;
    unsigned int tub;
    int ssd_dim;
    torch::Tensor output;
    std::vector<std::map<int, std::tuple<torch::Tensor, int, int>>> tubelets;
    std::vector<std::vector<std::pair<int, int>>> ides;
    std::vector<std::set<int>> ides_set;
    torch::Tensor history_max_ides;
    unsigned int hold_len;
//    unsigned feature_size;
    std::vector<cv::Scalar> color{cv::Scalar(255,255,0), cv::Scalar(0, 255,255), cv::Scalar(0,0,255), cv::Scalar(255,0,255)};
public:
    Detector(){};
    Detector(unsigned int, int, float, unsigned char, int);
    void init_detector(unsigned int, int, float, unsigned char, int);
    void log_params();
//    torch::Tensor detect(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
//            std::vector<float>, float, bool);
    void detect(const torch::Tensor&, const torch::Tensor&, std::vector<float>);
    void detect(const torch::Tensor&, const torch::Tensor&, std::vector<float>, float, bool);
    void visual_detect(const torch::Tensor&, const torch::Tensor&, std::vector<float>, float, bool, cv::Mat&, cv::VideoWriter&);
    std::tuple<torch::Tensor, int> nms(const torch::Tensor&, const torch::Tensor&);
    torch::Tensor iou(const torch::Tensor&, unsigned char);
    void visualization(cv::Mat&, cv::VideoWriter&);
    void init_tubelets();
    void delete_tubelets(unsigned char);
    void uart_send(unsigned char cls, Uart&);
    ~Detector(){
    };
};


#endif //RESDET_DETECTOR_H
