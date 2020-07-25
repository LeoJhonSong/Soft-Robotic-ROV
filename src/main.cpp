//
// Created by sean on 7/11/19.
//
#include "utils.h"
#include "detector.h"
#include "uart.h"
#include "ruas.h"
#include "rov.h"
#include "color.h"

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
#include<fstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <cstddef>


DEFINE_int32(K, 100, "turbulence intensity. The greater, the intensive");
DEFINE_int32(R, 40, "Signal to Noise Ratio. The greater, the more serious of noise");
DEFINE_uint32(RUAS, 0, "0: skip; 1: clahe; 2: wiener+clahe");
DEFINE_uint32(NET_PHASE, 2, "0: skip; 1: netG; 2: netG+RefineDet; 3: RefineDet" );
DEFINE_uint32(SSD_DIM, 320, "" );
DEFINE_uint32(NETG_DIM, 256, "" );
DEFINE_uint32(TUB, 0, "" );
DEFINE_int32(MODE, 0, "-1: load video; >0 load camera" );
DEFINE_bool(UART, false, "-1: not use it; >0 use it" );
DEFINE_bool(WITH_ROV, false, "0: not use it; >0 use it" );
DEFINE_bool(TRACK, false, "0: not use it; >0 use it" );


// for video_write thread
char EXT[] = "MJPG";
int ex1 = EXT[0] | (EXT[1] << 8) | (EXT[2] << 16) | (EXT[3] << 24);
time_t now = time(nullptr);
tm *ltm = localtime(&now);
std::string save_path =  std::to_string(1900 + ltm->tm_year) + "_" + std::to_string(1+ltm->tm_mon)+ "_" + std::to_string(ltm->tm_mday)
                         + "_" + std::to_string(ltm->tm_hour) + "_" + std::to_string(ltm->tm_min) + "_" + std::to_string(ltm->tm_sec);
cv::Size vis_size(640, 360);
bool save_a_frame = false;
bool save_a_count = false;
std::queue<cv::Mat> frame_queue, det_frame_queue;
std::queue<std::pair<cv::Mat, unsigned int>> img_queue;

int frame_w, frame_h;
bool video_write_flag = true;

// for run_rov thread
bool run_rov_flag = true;
int rov_key = 99;
bool rov_half_speed = false;
bool land = false;
int send_byte = -1;
unsigned char max_attempt = 0;
std::vector<int> target_loc = {0,0,0,0};
bool manual_stop = false;
bool grasping_done = false;
bool second_dive = false;
float max_depth = 0;
float curr_depth = 0;
float half_scale = 1.5;
float adjust_scale = 1.5;


int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    // make record dir and file
    if(nullptr==opendir(("./record/" + save_path).c_str()))
        mkdir(("./record/" + save_path).c_str(), S_IRWXU|S_IRWXG|S_IRWXO);
    std::ofstream log_file("./record/" + save_path + "/log.txt");

    // load models
    torch::NoGradGuard no_grad_guard;
    std::string model_path;

    if (FLAGS_NET_PHASE == 1) model_path = "./models/netG.pt";
    else if (FLAGS_NET_PHASE == 2) model_path = "./models/Unet256_SSD320_wof.pt";
    else if (FLAGS_SSD_DIM == 512) model_path = "./models/SSD512_wof.pt";
    else model_path = "./models/SSD320_wof.pt";

    torch::jit::script::Module net = torch::jit::load(model_path);
    net.to(at::kCUDA);

    // load detector
    unsigned int num_classes = 5;
    int top_k = 200;
    float nms_thresh = 0.3;
    std::vector<float> conf_thresh = {1.0, 0.8, 0.1, 1.5};
    float tub_thresh = 0.3;
    bool reset_id = false;
    Detector Detect(num_classes, top_k, nms_thresh, FLAGS_TUB, FLAGS_SSD_DIM, FLAGS_TRACK);

    // load filter
    CFilt filter(FLAGS_SSD_DIM, FLAGS_SSD_DIM, 3);
    if(FLAGS_RUAS>1) {
        filter.get_wf(FLAGS_K, FLAGS_R);
    }
    // load video
    cv::VideoCapture capture;
    while(!capture.isOpened()) {
        try{
            if (FLAGS_MODE == -1) {
//                capture.open("/home/sean/data/UWdevkit/snippets/echinus.mp4");
//                capture.open("/home/sean/Documents/ResDet/fine/2019_8_18_23_5_28/2019_8_18_23_5_28_raw.avi");
//                capture.set(cv::CAP_PROP_POS_FRAMES, 2700);
//                capture.open("/home/sean/Documents/ResDet/fine/Grab/2019_8_22_12_26_37_raw.avi");
//                capture.set(cv::CAP_PROP_POS_FRAMES, 13000);
                capture.open("/home/sean/Documents/ResDet/fine/FinalAutoGrab/2019_8_24_16_42_29_raw.avi");
                capture.set(cv::CAP_PROP_POS_FRAMES, 1100);
//                capture.open("/home/sean/Documents/ResDet/fine/OnlineDet/2019_8_22_12_54_48_raw.avi");
//                capture.set(cv::CAP_PROP_POS_FRAMES, 8000);
            } else if (FLAGS_MODE == -2) capture.open("rtsp://admin:zhifan518@192.168.1.88/11");
            else capture.open(FLAGS_MODE);
        }
        catch(const char* msg) {
            print(RED, "cannot open video");
            continue;
        }
    }
    frame_w = (int)capture.get(cv::CAP_PROP_FRAME_WIDTH);
    frame_h = (int)capture.get(cv::CAP_PROP_FRAME_HEIGHT);

    // intermediate variable
    cv::Mat frame, img_float, img_vis;
    torch::Tensor img_tensor, fake_B, loc, conf, ota_feature;
    std::vector<torch::jit::IValue> net_input, net_output;
    cv::cuda::GpuMat img_gpu;
    bool quit = false;
    unsigned char loc_idex = 0;
    time_t t_send = 0;
    int conut_times = 0;

    // UART
    Uart uart("ttyUSB0", 115200);
    if(FLAGS_UART) {
        bool uart_open_flag, uart_init_flag;
        uart_open_flag = uart.openFile();
        if (!uart_open_flag)
            print(BOLDRED, "ERROR: UART fails to open ");
        uart_init_flag = uart.initPort();
        if (!uart_init_flag)
            print(BOLDRED, "ERROR: UART fails to be initialed ");
    }


    // multi thread
    if (!FLAGS_WITH_ROV)
        run_rov_flag = false;
    std::thread rov_runner(run_rov);
    std::thread video_writer(video_write);

    while(capture.isOpened() && !quit){
        bool read_ret = capture.read(frame);
        if(!read_ret) break;
        frame_queue.push(frame);
        // pre processing
        cv::resize(frame, frame, cv::Size(FLAGS_NETG_DIM, FLAGS_NETG_DIM));
        if(FLAGS_RUAS == 1){
            filter.clahe_gpu(frame);
        }else if(FLAGS_RUAS == 2){
            filter.wiener_gpu(frame);
        }
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        // run net
        if (FLAGS_NET_PHASE > 0) {
            if (FLAGS_NET_PHASE < 3) {
                cv::normalize(frame, img_float, -1, 1, cv::NORM_MINMAX, CV_32F);
            } else if (FLAGS_NET_PHASE == 3) {
                frame.convertTo(img_float, CV_32F);
                img_float = img_float - 128.0;
            }
            img_tensor = torch::from_blob(img_float.data, {1, FLAGS_NETG_DIM, FLAGS_NETG_DIM, 3}).to(torch::kCUDA);
            img_tensor = img_tensor.permute({0, 3, 1, 2});
            net_input.emplace_back(img_tensor);
            if (FLAGS_NET_PHASE == 1) {
                fake_B = net.forward(net_input).toTensor();
                loc_idex = 1;
                cudaDeviceSynchronize();
            } else if (FLAGS_NET_PHASE > 1) {
                net_output = net.forward(net_input).toTuple()->elements();
                cudaDeviceSynchronize();
                if (FLAGS_NET_PHASE == 2) {
                    fake_B = net_output.at(0).toTensor();
                    loc_idex = 1;
                } else if (FLAGS_NET_PHASE == 3) loc_idex = 0;
                loc = net_output.at(loc_idex).toTensor().to(torch::kCPU);
                conf = net_output.at(loc_idex + 1).toTensor().to(torch::kCPU);
            }
            if (loc_idex==1) img_vis = tensor2im(fake_B);
            else img_vis = frame;
            net_input.pop_back();
            // detect
            cv::cvtColor(img_vis, img_vis, cv::COLOR_BGR2RGB);
            cv::resize(img_vis, img_vis, vis_size);
            target_loc = Detect.visual_detect(loc, conf, conf_thresh, tub_thresh, reset_id, img_vis, log_file);
//            print(BOLDRED, (float)target_loc[0]/vis_size.width << ", " << (float)target_loc[1]/vis_size.height << ", "<< (float)target_loc[2]/vis_size.width << ", " << (float)target_loc[3]/vis_size.height );
            if(land){
                if (FLAGS_UART) {
                    if (send_byte == -1) {
                        send_byte = Detect.uart_send(FLAGS_UART, uart);
                        print(BOLDCYAN, "MAIN: try to uart send, return " << send_byte);
                        if (send_byte == 6) {
                            print(BOLDCYAN, "MAIN: uart send successfully, clock start");
                            t_send = time(nullptr);
                        } else {
                            if (send_byte == 0) print(BOLDCYAN, "MAIN: uart fail to send");
                            else if (send_byte == 1){
                                print(BOLDCYAN, "MAIN: out of grasping area, try a second dive");
                                second_dive = true;
                            }
                            land = false;
                            grasping_done = true;
                            max_attempt = 0;
                            send_byte = -1;
                        }
                    }
                } else {
                    print(BOLDCYAN,  "MAIN: uart is closed");
                    land = false;
                    grasping_done = true;
                    max_attempt = 0;
                    send_byte = -1;
                }
            }
        }else{
            cv::cvtColor(frame.clone(), img_vis, cv::COLOR_BGR2RGB);
            cv::resize(img_vis, img_vis, vis_size);
            cv::imshow("ResDet", img_vis);
            det_frame_queue.push(img_vis);
            if (land) {
                print(BOLDCYAN, "MAIN: uart is closed");
                land = false;
                grasping_done = true;
                max_attempt = 0;
                send_byte = -1;
            }
        }
        if (send_byte == 6){
            if ((time(nullptr) - t_send) > 60) {
                send_byte = -1;
                if (++max_attempt>1) {
                    print(BOLDCYAN, "MAIN: max_attempt>2 grasping done");
                    land = false;
                    grasping_done = true;
                    max_attempt = 0;
                }
            }
        }
        int key = cv::waitKey(1);
        if (key != -1)  rov_key = key;
        parse_key(key, quit, reset_id, conf_thresh, FLAGS_K, FLAGS_R, filter);
        if (save_a_count) {
            print(BOLDWHITE, "save couting " + std::to_string(conut_times) + ": holothurian," << Detect.get_class_num(1) << ",echinus," << Detect.get_class_num(2)
                                                                                              << ",scallop," << Detect.get_class_num(3));
            std::ofstream result_file("./record/" + save_path + "/result_" + std::to_string(conut_times++) + ".txt");
            result_file << "Couting: holothurian," << Detect.get_class_num(1) << ",echinus," << Detect.get_class_num(2) << ",scallop," << Detect.get_class_num(3) << std::endl;
            result_file.close();
            save_a_count = false;

        }
    }
    print(BOLDWHITE, "save couting " + std::to_string(conut_times) + ": holothurian," << Detect.get_class_num(1) << ",echinus," << Detect.get_class_num(2)  << ",scallop," << Detect.get_class_num(3));
    std::ofstream result_file("./record/" + save_path + "/result_" + std::to_string(conut_times) + ".txt");
    result_file << "Couting: holothurian," << Detect.get_class_num(1) << ",echinus," << Detect.get_class_num(2)
                << ",scallop," << Detect.get_class_num(3) << std::endl;
    result_file.close();
    log_file.close();
    uart.closeFile();
    video_write_flag = false;
    video_writer.join();
    run_rov_flag = false;
    rov_runner.join();
    return 0;
}