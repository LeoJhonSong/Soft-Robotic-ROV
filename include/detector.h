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
    std::vector<std::set<int>> stable_ides_set;
    torch::Tensor history_max_ides;
    unsigned int hold_len;
    float large_size_filter;
    float small_size_filter;
    bool track;
    unsigned char track_cl;
    int track_id;
    unsigned int frame_num;
    std::vector<char> send_list;
    std::vector<cv::Scalar> color{cv::Scalar(255,0,0), cv::Scalar(255,255,0), cv::Scalar(0, 255,255), cv::Scalar(0,0,255), cv::Scalar(255,0,255)};
public:
    Detector(unsigned int, int, float, unsigned char, int, bool);
    void log_params();
    void detect(const torch::Tensor&, const torch::Tensor&, std::vector<float>);
    void detect(const torch::Tensor&, const torch::Tensor&, std::vector<float>, float);
    void detect_track(const torch::Tensor&, const torch::Tensor&, std::vector<float>);
    std::vector<int> visual_detect(const torch::Tensor&, const torch::Tensor&, const std::vector<float>&, float, bool&, cv::Mat&, std::ofstream&);
    std::tuple<torch::Tensor, int> nms(const torch::Tensor&, const torch::Tensor&);
    std::tuple<torch::Tensor, int> prev_nms(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&);
    torch::Tensor iou(const torch::Tensor&, unsigned char);
    std::vector<int> visualization(cv::Mat&, std::ofstream&);
    void init_tubelets();
    void delete_tubelets(unsigned char);
    void delete_tubelets();
    int uart_send(unsigned char cls, Uart&);
    void reset_tracking_state();
//    void enable_track();
//    void release_track();
    ~Detector();
};


#endif //RESDET_DETECTOR_H
