//
// Created by sean on 7/11/19.
//
#include "detector.h"
#include "ruas.h"
#include "color.h"
#include "marker_detector.h"
#include "parallel_camera.h"

#include <cuda_runtime.h>

#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <gflags/gflags.h>
#include <thread>
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <cstddef>


DEFINE_int32(K, 100, "turbulence intensity. The greater, the intensive");
DEFINE_int32(R, 50, "Signal to Noise Ratio. The greater, the more serious of noise");
DEFINE_uint32(RUAS, 0, "0: skip; 1: clahe; 2: wiener+clahe");
DEFINE_uint32(NET_PHASE, 2, "0: skip; 1: netG; 2: netG+RefineDet; 3: RefineDet" );
DEFINE_uint32(SSD_DIM, 320, "" );
DEFINE_uint32(NETG_DIM, 256, "" );
DEFINE_uint32(TUB, 1, "" );
DEFINE_int32(MODE, -1, "-2: load web camra; -1: load local video; >0: load camera" );
DEFINE_bool(TRACK, true, "0: not use it; >0 use it" );
DEFINE_bool(RECORD, false, "false: do not record raw and detected videos; true: record them");

// for video_write thread
char EXT[] = "MJPG";
int ex1 = EXT[0] | (EXT[1] << 8) | (EXT[2] << 16) | (EXT[3] << 24);
cv::Size vis_size(640, 360);
bool save_a_frame = false;
std::queue<cv::Mat> frame_queue, det_frame_queue;
std::queue<std::pair<cv::Mat, unsigned int>> img_queue;

int frame_w, frame_h;
bool video_write_flag = false;

// for run_rov thread
int send_byte = -1;
unsigned char max_attempt = 0;
std::vector<int> target_loc = {0,0,0,0};
float max_depth = 0;
float curr_depth = 0;
bool detect_scallop = false;

std::vector<int> target_info;
const int MARKER_OFFSET_X = 50;
const int MARKER_OFFSET_Y = 75;

// intermediate variable
cv::Mat frame, img_float, img_vis;
torch::Tensor img_tensor, fake_B, loc, conf, ota_feature;
std::vector<torch::jit::IValue> net_input, net_output;
cv::cuda::GpuMat img_gpu;
bool quit = false;
unsigned char loc_idex = 0;
int count_times = 0;

time_t now = time(nullptr);
tm *ltm = localtime(&now);
std::string save_path = std::to_string(1900 + ltm->tm_year) + "_" + std::to_string(1 + ltm->tm_mon) + "_" + std::to_string(ltm->tm_mday) + "_" + std::to_string(ltm->tm_hour) + "_" + std::to_string(ltm->tm_min) + "_" + std::to_string(ltm->tm_sec);
std::ofstream log_file("./record/" + save_path + "/log.txt");

cv::Mat tensor2im(torch::Tensor tensor) {
    tensor = tensor[0].add(1.0).div(2.0).mul(255.0).permute({1,2,0}).to(torch::kU8).to(torch::kCPU);
    cv::Mat img(tensor.size(0), tensor.size(1), CV_8UC3);
    std::memcpy((void*)img.data, tensor.data_ptr(), sizeof(torch::kU8)*tensor.numel());
    return img;
}

void video_write(){
    // 如果不录制视频, 退出视频录制线程
    if (!video_write_flag) return;
    //raw video
    cv::VideoWriter writer_raw;
    writer_raw.open("./record/" + save_path + "/" + save_path + "_raw.avi", ex1, 20, cv::Size(frame_w, frame_h), true);
    if(!writer_raw.isOpened()){
        print(BOLDRED, "ERROR: Can not open the output video for raw write");
    }
    //det video
    cv::VideoWriter writer_det;
    writer_det.open("./record/" + save_path + "/" + save_path + "_det.avi", ex1, 20, vis_size, true);
    if(!writer_det.isOpened()){
        print(BOLDRED, "ERROR: Can not open the output video for det write");
    }

    while(video_write_flag) {
        if (!frame_queue.empty()) {
            writer_raw << frame_queue.front();
            frame_queue.pop();
        }
        if (!det_frame_queue.empty()) {
            writer_det << det_frame_queue.front();
            det_frame_queue.pop();
        }
        if (!img_queue.empty()) {
            cv::imwrite("./record/" + save_path + "/" + std::to_string(img_queue.front().second) + ".jpg", img_queue.front().first);
            img_queue.pop();
        }
    }
    print(RED, "QUIT: video write thread quit");
    writer_raw.release();
    writer_det.release();
}

int main(int argc, char* argv[]) {
    time_t now = std::time(0);
    char* date = std::ctime(&now);
    print(BOLDGREEN, "starting at " << date);
    // 读入命令行参数
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    // 设置是否录制
    video_write_flag = FLAGS_RECORD;
    // make record dir and file
    if (FLAGS_RECORD)
    {
        if (nullptr == opendir(("./record/" + save_path).c_str()))
            mkdir(("./record/" + save_path).c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
    }
    // load models
    torch::NoGradGuard no_grad_guard;
    std::string model_path;

    if (FLAGS_NET_PHASE == 1) model_path = "./models/netG.pt";
    else if (FLAGS_NET_PHASE == 2) model_path = "./models/Unet256_SSD320_wof.pt";
    else if (FLAGS_SSD_DIM == 512) model_path = "./models/SSD512_wof.pt";
    else model_path = "./models/SSD320_wof.pt";

    std::shared_ptr<torch::jit::script::Module> net = torch::jit::load(model_path);
    net->to(at::kCUDA);


    // load detector
    unsigned int num_classes = 5;
    int top_k = 200;
    float nms_thresh = 0.3;
    std::vector<float> conf_thresh = {0.6, 0.8, 0.3, 1.5};  // 海参, 海胆, 扇贝, 海星
    float tub_thresh = 0.3;
    bool reset_id = false;
    Detector Detect(num_classes, top_k, nms_thresh, FLAGS_TUB, FLAGS_SSD_DIM, FLAGS_TRACK);

    // load filter
    CFilt filter(FLAGS_SSD_DIM, FLAGS_SSD_DIM, 3);
    if (FLAGS_RUAS > 1)
    {
        filter.get_wf(FLAGS_K, FLAGS_R);
    }

    // load video
    ParallelCamera capture;
    while (!capture.isOpened())
    {
        try
        {
            if (FLAGS_MODE == -1)
            {

                capture.open("./test/echinus.mp4");
                // 设置从视频的哪一帧开始读取
                capture.set(cv::CAP_PROP_POS_FRAMES, 1100);
            }
            else if (FLAGS_MODE == -2)
                capture.open("rtsp://admin:zhifan518@192.168.1.88/11");
            else
                capture.open(FLAGS_MODE);
        }
        catch (const char *msg)
        {
            print(RED, "cannot open video");
            continue;
        }
    }
    frame_w = (int)capture.get(cv::CAP_PROP_FRAME_WIDTH);
    frame_h = (int)capture.get(cv::CAP_PROP_FRAME_HEIGHT);

    std::thread video_writer(video_write);
    capture.receive_start();  // 视频流读取线程

    // marker detector
    // 初始化的size要对应上后面输入图片的size,看到时候用哪个图片(原始的frame, net_G输出的fake_B, 或者resize后的img_vis)比较好
	// marker::MarkerDetector marker_detector(frame.size());
	marker::MarkerDetector marker_detector(vis_size);
	marker::MarkerInfo marker_info_current;
	marker::MarkerInfo marker_info;

    while(capture.isOpened() && !quit){
        // 获取视频流中最新帧
        bool read_ret = capture.read(frame);
        if(!read_ret) break;
        // pre processing
        cv::resize(frame, frame, cv::Size(FLAGS_NETG_DIM, FLAGS_NETG_DIM));
        if(FLAGS_RUAS == 1){
            filter.clahe_gpu(frame);
        }else if(FLAGS_RUAS == 2){
            filter.wiener_gpu(frame);
        }
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        // run net
        if (FLAGS_NET_PHASE < 3)
        {
            cv::normalize(frame, img_float, -1, 1, cv::NORM_MINMAX, CV_32F);
        }
        else if (FLAGS_NET_PHASE == 3)
        {
            frame.convertTo(img_float, CV_32F);
            img_float = img_float - 128.0;
        }
        img_tensor = torch::from_blob(img_float.data, {1, FLAGS_NETG_DIM, FLAGS_NETG_DIM, 3}).to(torch::kCUDA);
        img_tensor = img_tensor.permute({0, 3, 1, 2});
        net_input.emplace_back(img_tensor);
        if (FLAGS_NET_PHASE == 1)
        {
            fake_B = net->forward(net_input).toTensor();
            loc_idex = 1;
            cudaDeviceSynchronize();
        }
        else if (FLAGS_NET_PHASE > 1)
        {
            net_output = net->forward(net_input).toTuple()->elements();
            cudaDeviceSynchronize();
            if (FLAGS_NET_PHASE == 2)
            {
                fake_B = net_output.at(0).toTensor();
                loc_idex = 1;
            }
            else if (FLAGS_NET_PHASE == 3)
                loc_idex = 0;
            loc = net_output.at(loc_idex).toTensor().to(torch::kCPU);
            conf = net_output.at(loc_idex + 1).toTensor().to(torch::kCPU);
        }
        if (loc_idex == 1)
            img_vis = tensor2im(fake_B);
        else
            img_vis = frame;
        net_input.pop_back();
        // detect
        cv::cvtColor(img_vis, img_vis, cv::COLOR_BGR2RGB);
        cv::resize(img_vis, img_vis, vis_size);

        // detect marker
        marker_info_current = marker_detector.detect_single_marker(img_vis, true, marker::VER_OPENCV, marker::MODE_DETECT);
        if (marker_info_current.center.x > 0 && marker_info_current.center.y > 0)
        {
            marker_info = marker_info_current;
            // 补偿偏置
            marker_info.center.x += MARKER_OFFSET_X;
            marker_info.center.y += MARKER_OFFSET_Y;
        }
        // 补偿后原点
        cv::circle(img_vis, marker_info.center, 6, cv::Scalar(0, 0, 255), -1, 8, 0);
        // print(BOLDYELLOW, "x: " << marker_info.center.x << " y: " << marker_info.center.y);

        target_loc = Detect.visual_detect(loc, conf, conf_thresh, tub_thresh, reset_id, img_vis, log_file);
        // print(BOLDRED, (float)target_loc[0]/vis_size.width << ", " << (float)target_loc[1]/vis_size.height << ", "<< (float)target_loc[2]/vis_size.width << ", " << (float)target_loc[3]/vis_size.height );
        if((cv::waitKey(1) & 0xFF) == 27) quit = true;
    }
    print(BOLDWHITE, "save couting " + std::to_string(count_times) + ": holothurian," << Detect.get_class_num(1) << ",echinus," << Detect.get_class_num(2)  << ",scallop," << Detect.get_class_num(3));
    log_file.close();
    video_write_flag = false;
    video_writer.join();
    capture.receive_stop();
    print(BOLDGREEN, "bye!");
    return 0;
}
