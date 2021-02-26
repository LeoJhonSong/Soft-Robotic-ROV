#ifndef RESDET_DETECTOR_H
#define RESDET_DETECTOR_H

#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

namespace detector
{
struct Visual_info
{
    // TODO: (optional) add target_time, marker_time?
    // for target
    bool has_target;
    unsigned int target_class;
    int target_id;
    // float target_confidence = 0;
    cv::Point2f target_center; // cx, cy
    cv::Point2f target_shape;  // width, height
    // for arm
    bool arm_is_working; // from outside
    bool has_marker;
    cv::Point2f marker_position;
    Visual_info()
    {
        has_target = false;
        target_class = 0;
        target_id = -1;
        target_center = cv::Point2f(0, 0);
        target_shape = cv::Point2f(0, 0);
        arm_is_working = false;
        has_marker = false;
        marker_position = cv::Point2f(0, 0);
    }
};

class Detector
{
  private:
    unsigned int num_classes;
    int top_k;
    float nms_thresh;
    bool tub;
    int ssd_dim;
    torch::Tensor candidates;
    std::vector<std::map<int, std::tuple<torch::Tensor, int, int>>> tubelets;
    std::vector<std::vector<std::pair<int, int>>> ides;
    std::vector<std::set<int>> ides_set;
    std::vector<std::set<int>> stable_ides_set;
    torch::Tensor history_max_ides;
    unsigned int hold_len;
    float large_size_filter;
    float small_size_filter;
    float y_max_filter;
    bool track;
    std::vector<cv::Scalar> color{cv::Scalar(255, 0, 0), cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 255),
                                  cv::Scalar(0, 0, 255), cv::Scalar(255, 0, 255)};

  public:
    unsigned int tracking_class;
    int track_id;
    Detector(unsigned int, int, float, bool, int, bool);
    void log_params();
    int get_class_num(unsigned char);
    std::tuple<torch::Tensor, int> nms(torch::Tensor &, torch::Tensor &);
    std::tuple<torch::Tensor, int> prev_nms(torch::Tensor &, torch::Tensor &, const torch::Tensor &);
    torch::Tensor iou(const torch::Tensor &, unsigned char);
    void init_tubelets();
    void delete_tubelets(unsigned char);
    void delete_tubelets();
    void replenish_tubelets(unsigned char cl, int count);
    void reset_tracking_state();
    void update(const torch::Tensor &loc, const torch::Tensor &conf, std::vector<float> conf_thresh);
    void update(const torch::Tensor &loc, const torch::Tensor &conf, std::vector<float> conf_thresh, float tub_thresh);
    void tracking_update(const torch::Tensor &loc, const torch::Tensor &conf, std::vector<float> conf_thresh);
    std::vector<float> visualization(cv::Mat &);
    std::vector<float> detect_and_visualize(const torch::Tensor &, const torch::Tensor &, const std::vector<float> &,
                                            float, bool &, bool, cv::Mat &);
    ~Detector();
};
} // namespace detector

#endif // RESDET_DETECTOR_H
