//
// Created by sean on 7/11/19.
//
#include "utils.h"
#include "detector.h"
#include "uart.h"
#include "ruas.h"
#include "rov.h"

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

DEFINE_int32(K, 100, "turbulence intensity. The greater, the intensive");
DEFINE_int32(R, 40, "Signal to Noise Ratio. The greater, the more serious of noise");
DEFINE_uint32(RUAS, 0, "0: skip; 1: clahe; 2: wiener+clahe");
DEFINE_uint32(NET_PHASE, 0, "0: skip; 1: netG; 2: netG+RefineDet; 3: RefineDet" );
DEFINE_uint32(SSD_DIM, 320, "" );
DEFINE_uint32(NETG_DIM, 320, "" );
DEFINE_uint32(TUB, 0, "" );
DEFINE_int32(MODE, -1, "-1: load video; >0 load camera" );
DEFINE_int32(UART, 0, "-1: not use it; >0 use it" );
DEFINE_int32(WITH_ROV, 0, "0: not use it; >0 use it" );

char EXT[] = "MJPG";
int ex1 = EXT[0] | (EXT[1] << 8) | (EXT[2] << 16) | (EXT[3] << 24);
time_t now = time(nullptr);
tm *ltm = localtime(&now);
std::string video_name = "./record/" + std::to_string(1900 + ltm->tm_year) + "_" + std::to_string(1+ltm->tm_mon)+ "_" + std::to_string(ltm->tm_mday)
                         + "_" + std::to_string(ltm->tm_hour) + "_" + std::to_string(ltm->tm_min) + "_" + std::to_string(ltm->tm_sec);

// run_rov thread
bool run_rov_flag = true;
int rov_key;

// raw_write thread
std::queue<cv::Mat> frame_queue;
int frame_w, frame_h;
bool raw_write_flag = true;

// run_net thread
std::queue<cv::Mat> processed_frame;
std::queue<std::tuple<cv::Mat, torch::Tensor, torch::Tensor>> from_net;
bool run_net_flag = true;
unsigned char loc_idex = 0;
void run_net() {
    // load models
    torch::NoGradGuard no_grad_guard;
    std::string model_path;

    if (FLAGS_NET_PHASE == 1) model_path = "./models/netG.pt";
    else if (FLAGS_NET_PHASE == 2) model_path = "./models/Unet256_SSD320_wof.pt";
    else if (FLAGS_SSD_DIM == 512) model_path = "./models/SSD512_wof.pt";
    else model_path = "./models/SSD320_wof.pt";

    std::shared_ptr<torch::jit::script::Module> net = torch::jit::load(model_path);
    net->to(at::kCUDA);
    cv::Mat img_float, img_vis;
    torch::Tensor img_tensor, fake_B, loc, conf;
    std::vector<torch::jit::IValue> net_input, net_output;
    while (run_net_flag) {
        if (!processed_frame.empty()) {
            if (FLAGS_NET_PHASE < 3) {
                cv::normalize(processed_frame.front(), img_float, -1, 1, cv::NORM_MINMAX, CV_32F);
            } else if (FLAGS_NET_PHASE == 3) {
                processed_frame.front().convertTo(img_float, CV_32F);
                img_float = img_float - 128.0;
            }
            img_tensor = torch::from_blob(img_float.data, {1, FLAGS_NETG_DIM, FLAGS_NETG_DIM, 3}).to(torch::kCUDA);
            img_tensor = img_tensor.permute({0, 3, 1, 2});
            net_input.emplace_back(img_tensor);
            if (FLAGS_NET_PHASE == 1) {
                fake_B = net->forward(net_input).toTensor();
                loc_idex = 1;
                cudaDeviceSynchronize();
            } else if (FLAGS_NET_PHASE > 1) {
                net_output = net->forward(net_input).toTuple()->elements();
                cudaDeviceSynchronize();
                if (FLAGS_NET_PHASE == 2) {
                    fake_B = net_output.at(0).toTensor();
                    loc_idex = 1;
                } else if (FLAGS_NET_PHASE == 3) loc_idex = 0;
                loc = net_output.at(loc_idex).toTensor().to(torch::kCPU);
                conf = net_output.at(loc_idex + 1).toTensor().to(torch::kCPU);
            }
            if (loc_idex==1) img_vis = tensor2im(fake_B);
            else img_vis = processed_frame.front();
            net_input.pop_back();
            from_net.push(std::tuple<cv::Mat, torch::Tensor, torch::Tensor>{img_vis, loc, conf});
            processed_frame.pop();
        }

    }
    std::cout << "run_net thread qiut" << std::endl;
}

int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // load detector
    unsigned int num_classes = 5;
    int top_k = 200;
    float nms_thresh = 0.3;
    std::vector<float> conf_thresh = {0.3, 0.3, 0.5, 0.5};
    float tub_thresh = 0.1;
    bool reset_id = false;
    Detector Detect(num_classes, top_k, nms_thresh, FLAGS_TUB, FLAGS_SSD_DIM);

    // load filter
    CFilt filter(FLAGS_SSD_DIM, FLAGS_SSD_DIM, 3);
    if(FLAGS_RUAS>1) {
        filter.get_wf(FLAGS_K, FLAGS_R);
    }

    // load video
    cv::VideoCapture capture;
    if(FLAGS_MODE == -1){
        capture.open("/home/sean/data/UWdevkit/snippets/echinus.mp4");
//        capture.set(CV_CAP_PROP_POS_FRAMES, 200);
    }
    else if(FLAGS_MODE == -2) capture.open("rtsp://admin:zhifan518@192.168.1.88/11");
    else capture.open(FLAGS_MODE);
    frame_w = (int)capture.get(CV_CAP_PROP_FRAME_WIDTH);
    frame_h = (int)capture.get(CV_CAP_PROP_FRAME_HEIGHT);
    cv::Size vis_size(640, 360);

    //out video
    cv::VideoWriter writer;
    writer.open(video_name + ".mp4", ex1, 25, vis_size, true);
    if(!writer.isOpened()){
        std::cout << "Can not open the output video for write" << std::endl;
    }

    // intermediate variable
    cv::Mat frame, img_float, img_vis;
    torch::Tensor img_tensor, fake_B, loc, conf, ota_feature;
    std::vector<torch::jit::IValue> net_input, net_output;
    cv::cuda::GpuMat img_gpu;
    Uart uart("ttyUSB0", 115200);

    // UART
    if(FLAGS_UART) {
        bool uart_open_flag, uart_init_flag;
        uart_open_flag = uart.openFile();
        if (!uart_open_flag)
            std::cout << "UART fails to open " << std::endl;
        uart_init_flag = uart.initPort();
        if (!uart_init_flag)
            std::cout << "UART fails to be inited " << std::endl;
    }

    bool quit = false;
    clock_t t_start= clock();
    unsigned int frame_num = 0;
    int wait_mm = 1;
    if (FLAGS_TUB == 0 && FLAGS_MODE==-1) wait_mm = 10;

    // multi thread
    if (FLAGS_NET_PHASE == 0)
        run_net_flag = false;
    std::thread net_runner(run_net);
    if (FLAGS_WITH_ROV == 0)
        run_rov_flag = false;
    std::thread rov_runner(run_rov);
    std::thread raw_writer(raw_write);


    while(capture.isOpened() && !quit){
        bool read_ret = capture.read(frame);
        if(!read_ret) break;
        frame_num ++;
        frame_queue.push(frame);
        cv::resize(frame, frame, cv::Size(FLAGS_NETG_DIM, FLAGS_NETG_DIM));
        if(FLAGS_RUAS == 1){
            filter.clahe_gpu(frame);
        }else if(FLAGS_RUAS == 2){
            filter.wiener_gpu(frame);
        }
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        processed_frame.push(frame);
//        std::cout << "main, processed_frame size: " << processed_frame.size() << std::endl;
        if (processed_frame.size()>=5) processed_frame.pop();

        if (!from_net.empty()) {
            cv::cvtColor(std::get<0>(from_net.front()), img_vis, cv::COLOR_BGR2RGB);
            cv::resize(img_vis, img_vis, vis_size);

            Detect.visual_detect(std::get<1>(from_net.front()), std::get<2>(from_net.front()), conf_thresh, tub_thresh, reset_id, img_vis, writer);
            if (reset_id) reset_id = false;
            if (FLAGS_UART > 0)
                Detect.uart_send(FLAGS_UART, uart);
            from_net.pop();
        }
        else if (FLAGS_NET_PHASE==0){
            cv::cvtColor(frame.clone(), img_vis, cv::COLOR_BGR2RGB);
            cv::resize(img_vis, img_vis, vis_size);
            cv::imshow("ResDet", img_vis);
            writer << img_vis;
        }
        int key = cv::waitKey(wait_mm);
        rov_key = key;
//        std::cout << key << std::endl;
        parse_key(key, quit, reset_id, conf_thresh, FLAGS_K, FLAGS_R, filter);
//        std::cout << "total: " << (t2 - t1) * 1.0 / CLOCKS_PER_SEC * 1000
//                  << ", ruas: " << (t6 - t1) * 1.0 / CLOCKS_PER_SEC * 1000
//                  << ", vis: " << (t7 - t2) * 1.0 / CLOCKS_PER_SEC * 1000 << std::endl;

    }
    clock_t t_end= clock();
    std::cout << "frame amount: " << frame_num  << ", FPS: " << (double)frame_num / ((t_end - t_start) * 1.0 / CLOCKS_PER_SEC) << std::endl;

    uart.closeFile();
    writer.release();
    raw_write_flag = false;
    raw_writer.join();
    run_net_flag = false;
    net_runner.join();
    run_rov_flag = false;
    rov_runner.join();
    writer.release();
    return 0;
}
