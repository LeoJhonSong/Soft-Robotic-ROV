#ifndef PARALLEL_CAMETA_H
#define PARALLEL_CAMETA_H

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <thread>
void test();
class ParallelCamera: public cv::VideoCapture
{
private:
    cv::Mat current_frame;
    cv::Mat testf;
    bool current_read_ret;
    bool is_running = false;
    std::thread* receive_thread = nullptr;
    void receive();
public:
    bool read(cv::Mat &image);
    void receive_start();
    void receive_stop();
};

#endif