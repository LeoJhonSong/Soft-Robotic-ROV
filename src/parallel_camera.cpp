#include "parallel_camera.h"
#include "color.h"
#include <functional>
#include <unistd.h>

extern std::queue<cv::Mat> frame_queue;
extern bool FLAGS_BINOCULAR;
const int CAPTURE_TYPE_CAMERA = 1;
const int CAPTURE_TYPE_VIDEO = 2;

void ParallelCamera::receive()
{
    while (this->is_running)
    {
        if (this->video_type == CAPTURE_TYPE_VIDEO)
        {
            // 限制本地视频读取速度. 延时1秒/FPS的一半
            usleep(1000000.0 / this->get(cv::CAP_PROP_FPS) * 0.5);
        }
        this->current_read_ret = this->cv::VideoCapture::read(this->current_frame);
        if (!this->current_read_ret)
        {
            this->is_running = false;
            continue;
        }
        frame_queue.push(this->current_frame);
        // cv::waitKey(1);
        // cv::imshow("frame", this->current_frame);
    }
}

// copy image read by ParallelCamera::receive()
bool ParallelCamera::read(cv::Mat &image)
{
    if (FLAGS_BINOCULAR)
    {
        cv::Rect rect(0, 0, int(this->current_frame.cols / 2), this->current_frame.rows); // x, y of top left + width, height
        image = this->current_frame(rect);
    }
    else
    {
        image = this->current_frame.clone();  // able to have 竞争冒险 if not copy
    }
    return current_read_ret;
}

bool ParallelCamera::open(const cv::String &filename)
{
    if (filename.find_last_of("mp4") != cv::String::npos ||
        filename.find_last_of("MP4") != cv::String::npos ||
        filename.find_last_of("avi") != cv::String::npos ||
        filename.find_last_of("AVI") != cv::String::npos)
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
        this->receive_thread = new std::thread(std::mem_fn(&ParallelCamera::receive), this);
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
        this->receive_thread = nullptr;
    }
}
