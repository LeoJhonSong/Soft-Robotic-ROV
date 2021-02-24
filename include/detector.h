#ifndef RESDET_DETECTOR_H
#define RESDET_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

namespace detector
{
    struct Visual_info
    {
        // TODO: (optional) add target_time, marker_time?
        // for target
        bool has_target = false;
        unsigned int target_class = 0;
        int target_id = -1;
        // float target_confidence = 0;
        cv::Point2f target_center = cv::Point2f(0, 0); // cx, cy
        cv::Point2f target_shape = cv::Point2f(0, 0);  //width, height
        // for arm
        bool arm_is_working = false;  // from outside
        bool has_marker = false;
        cv::Point2f marker_position = cv::Point2f(0, 0);
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
        std::vector<cv::Scalar> color{cv::Scalar(255, 0, 0), cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(0, 0, 255), cv::Scalar(255, 0, 255)};

    public:
        unsigned int tracking_class;
        int track_id;
        Detector(unsigned int, int, float, bool, int, bool);
        void log_params();
        void update(const torch::Tensor &loc, const torch::Tensor &conf, std::vector<float> conf_thresh);
        void update(const torch::Tensor &loc, const torch::Tensor &conf, std::vector<float> conf_thresh, float tub_thresh);
        void tracking_update(const torch::Tensor &loc, const torch::Tensor &conf, std::vector<float> conf_thresh);
        std::vector<float> detect_and_visualize(const torch::Tensor &, const torch::Tensor &, const std::vector<float> &, float, bool &, cv::Mat &);
        std::tuple<torch::Tensor, int> nms(torch::Tensor &, torch::Tensor &);
        std::tuple<torch::Tensor, int> prev_nms(torch::Tensor &, torch::Tensor &, const torch::Tensor &);
        torch::Tensor iou(const torch::Tensor &, unsigned char);
        std::vector<float> visualization(cv::Mat &);
        void init_tubelets();
        void delete_tubelets(unsigned char);
        void delete_tubelets();
        void get_relative_position();
        void reset_tracking_state();
        int get_class_num(unsigned char);
        void replenish_tubelets(unsigned char cl, int count);
        ~Detector();
    };
} // namespace detector

#endif //RESDET_DETECTOR_H
