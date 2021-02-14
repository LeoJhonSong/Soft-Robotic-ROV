#ifndef PARALLEL_CAMETA_H
#define PARALLEL_CAMETA_H

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <thread>
class ParallelCamera : public cv::VideoCapture
{
private:
    cv::Mat current_frame;
    cv::Mat testf;
    bool current_read_ret;
    bool is_running = false;
    int video_type = 0;
    std::thread *receive_thread = nullptr;
    void receive();

public:
    bool read(cv::Mat &image);
    bool open(const cv::String &filename);
    bool open(int index);
    void receive_start();
    void receive_stop();
};

#endif