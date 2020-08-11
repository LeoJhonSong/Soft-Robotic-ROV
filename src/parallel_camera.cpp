#include "parallel_camera.h"
#include "color.h"
#include <functional> 

extern std::queue<cv::Mat> frame_queue;

void test()
{
    print(RED, 233);
    cv::waitKey(1);
}

void ParallelCamera::receive()
{
    while (this->is_running)
    {
        this->current_read_ret = this->cv::VideoCapture::read(this->current_frame);
        frame_queue.push(this->current_frame);
    }
}

bool ParallelCamera::read(cv::Mat &image)
{
    image = this->current_frame.clone();
    return current_read_ret;
}

void ParallelCamera::receive_start()
{
    if (this->receive_thread == nullptr)
    {
        this->current_read_ret = this->cv::VideoCapture::read(this->current_frame);
        receive_thread = new std::thread(std::mem_fn(&ParallelCamera::receive), this);
        this->is_running = true;
    }
    else
    {
        print(RED, "The receive thread has run!");
    }
}

void ParallelCamera::receive_stop()
{
    if (this->receive_thread == nullptr)
    {
        print(RED, "The receive thread is not running!");
    }
    else
    {
        this->is_running = false;
        (*(this->receive_thread)).join();
        this->receive_thread == nullptr;
    }
}
