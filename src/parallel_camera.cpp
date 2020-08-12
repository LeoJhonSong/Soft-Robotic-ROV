#include "parallel_camera.h"
#include "color.h"
#include <unistd.h>
#include <functional> 

extern std::queue<cv::Mat> frame_queue;
const int CAPTURE_TYPE_NULL = 0;
const int CAPTURE_TYPE_CAMERA = 1;
const int CAPTURE_TYPE_VIDEO = 2;

void test()
{
    print(RED, 233);
    cv::waitKey(1);
}

void ParallelCamera::receive()
{
    while (this->is_running)
    {
        if (this->video_type == CAPTURE_TYPE_VIDEO)
        {
            cv::waitKey(1);
            // 延时 微秒
            usleep(1000000.0 / this->get(cv::CAP_PROP_FPS) * 0.5);
        }
        this->current_read_ret = this->cv::VideoCapture::read(this->current_frame);
        if (!this->current_read_ret)
        {
            this->is_running = false;
            continue;
        }
        frame_queue.push(this->current_frame);
        cv::imshow("frame", this->current_frame);
    }
}

bool ParallelCamera::read(cv::Mat &image)
{
    image = this->current_frame.clone();
    return current_read_ret;
}

bool ParallelCamera::open(const cv::String &filename)
{
    if (filename.find_last_of(".mp4") != cv::String::npos ||
        filename.find_last_of(".MP4") != cv::String::npos ||
        filename.find_last_of(".avi") != cv::String::npos ||
        filename.find_last_of(".AVI") != cv::String::npos)
    {
        this->video_type = CAPTURE_TYPE_VIDEO;
    }
    else
    {
        this->video_type = CAPTURE_TYPE_CAMERA;
    }
    return this->cv::VideoCapture::open(filename);
}

bool ParallelCamera::open(int index)
{
    this->video_type = CAPTURE_TYPE_CAMERA;
    return this->cv::VideoCapture::open(index);
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
