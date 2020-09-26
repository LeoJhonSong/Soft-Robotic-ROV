//
// Created by sean on 7/11/19.
//
#include "utils.h"
#include "detector.h"
#include "uart.h"
#include "ruas.h"
#include "rov.h"
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
DEFINE_bool(UART, false, "false: do not try to communicate by UART; true: try to communicate by UART" );
DEFINE_bool(WITH_ROV, false, "false: do not try to connect ROV; true: try to connect to ROV" );
DEFINE_bool(TRACK, true, "0: not use it; >0 use it" );
DEFINE_bool(RECORD, false, "false: do not record raw and detected videos; true: record them");

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
bool video_write_flag = false;

// for run_rov thread
bool run_rov_flag = true;
int rov_key = 32;
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
bool detect_scallop = true;

std::vector<int> target_info;
const int TIME_PER_GRAP = 70;
const int MARKER_OFFSET_X = 50;
const int MARKER_OFFSET_Y = 75;
extern const float GRAP_THRESH_XC = 0.5;
extern const float GRAP_THRESH_XW = 0.2;
extern const float GRAP_THRESH_YC = 0.75;
extern const float GRAP_THRESH_YW = 0.15;

int main(int argc, char* argv[]) {
    time_t now = std::time(0);
    char* date = std::ctime(&now);
    print(BOLDGREEN, "starting at " << date);
    // 读入命令行参数
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    // 设置是否录制
    video_write_flag = FLAGS_RECORD;
    // make record dir and file
    if(FLAGS_RECORD)
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

    std::shared_ptr<torch::jit::script::Module> net = torch::jit::load(model_path);
    net->to(at::kCUDA);


    // load detector
    unsigned int num_classes = 5;
    int top_k = 200;
    float nms_thresh = 0.3;
    std::vector<float> conf_thresh = {1.5, 1.8, 0.1, 1.5};  // 海参, 海胆, 扇贝, 海星
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
    Uart uart("ttyUSB0", 9600);
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
    capture.receive_start();  // 视频流读取线程

    // marker detector
    // 初始化的size要对应上后面输入图片的size,看到时候用哪个图片(原始的frame, net_G输出的fake_B, 或者resize后的img_vis)比较好
    // capture.read(frame);
	// marker::MarkerDetector marker_detector(frame.size());
	marker::MarkerDetector marker_detector(vis_size);
	marker::MarkerInfo marker_info_current;
	marker::MarkerInfo marker_info;
    int i = 0;
    int key_1 = -1;
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
        // marker_info_current = marker_detector.detect_single_marker(img_vis, true, marker::VER_OPENCV, marker::MODE_DETECT);
        marker_info_current = marker_detector.detect_average_marker(img_vis, true, marker::VER_OPENCV, marker::MODE_DETECT);
        if (marker_info_current.center.x > 0 && marker_info_current.center.y > 0)
        {
            marker_info = marker_info_current;
            // // 补偿偏置
            // marker_info.center.x += MARKER_OFFSET_X;
            // marker_info.center.y += MARKER_OFFSET_Y;
        }
        // 补偿后原点
        cv::circle(img_vis, marker_info.center, 6, cv::Scalar(0, 0, 255), -1, 8, 0);
        // print(BOLDYELLOW, "x: " << marker_info.center.x << " y: " << marker_info.center.y);
        cv::rectangle(
            img_vis, 
            cv::Point2f(vis_size.width*(GRAP_THRESH_XC-GRAP_THRESH_XW), vis_size.height*(GRAP_THRESH_YC-GRAP_THRESH_YW)),
            cv::Point2f(vis_size.width*(GRAP_THRESH_XC+GRAP_THRESH_XW), vis_size.height*(GRAP_THRESH_YC+GRAP_THRESH_YW)),
            cv::Scalar(0, 0, 255),
            2
        );
        target_loc = Detect.visual_detect(loc, conf, conf_thresh, tub_thresh, reset_id, img_vis, log_file);
        // print(BOLDRED, (float)target_loc[0]/vis_size.width << ", " << (float)target_loc[1]/vis_size.height << ", "<< (float)target_loc[2]/vis_size.width << ", " << (float)target_loc[3]/vis_size.height );
        
        // 半自主时 按空格开始抓取
        if (!FLAGS_WITH_ROV)
        {
            if (key_1 == ' ')
            {
                land = true;
            }
        }
        // 坐底后的串口通信
        if (land && ((!grasping_done) || (!FLAGS_WITH_ROV)))//
        {
            if (FLAGS_UART)
            {
                if (send_byte == -1)
                {
                    print(RED, "target_loc: " << target_loc);
                    target_info = Detect.get_relative_position(uart);
                    print(RED, "target_info: " << target_info);
                    if(target_info[0] == 1000)
                    {
                        send_byte = 0;
                        uart.send("No target");
                    }
                    else if(target_info[0] == 2000)
                    {
                    // else if(false)
                        send_byte = 1;
                        uart.send("No target");
                    }
                    else
                    {
                        // 全自动抓取，叹号代表抓取开始信号
                        std::string send_array = "!!";
                        uart.send(send_array);
                        // 发送marker相对于目标的坐标
                        send_array = "#";
                        send_array = send_array + std::to_string(-(marker_info.center.x / vis_size.width * 100 - target_info[1])) + "," + 
                                    std::to_string(-(marker_info.center.y / vis_size.height * 100 - target_info[2]))+ "," + 
                                    std::to_string(curr_depth) + 
                                    "\n";
                        print(BOLDYELLOW, "t_x: " << target_info[1] << " t_y: " << target_info[2]);

                        send_byte = uart.send(send_array);

                        print(BOLDGREEN, "[marker relative position] " << send_array);
                    }

                    if (send_byte == 6)
                    {
                        print(BOLDCYAN, "MAIN: uart send successfully, clock start");
                        t_send = time(nullptr);
                    }
                    else
                    {
                        if (send_byte == 0)
                            print(BOLDCYAN, "MAIN: uart fail to send");
                        else if (send_byte == 1)
                        {
                            print(BOLDCYAN, "MAIN: out of grasping area, try a second dive");
                            // 这是废的
                            second_dive = true;
                        }
                        land = false;
                        grasping_done = true;
                        max_attempt = 0;
                        send_byte = -1;
                    }
                }
            }
            else
            {
                print(BOLDCYAN, "MAIN: uart is closed");
                land = false;
                grasping_done = true;
                max_attempt = 0;
                send_byte = -1;
            }
        }
        // 串口通信成功后
        if (send_byte == 6)
        {
            if ((time(nullptr) - t_send) > TIME_PER_GRAP)  // 时间超过180s, 判断再尝试一次还是放弃当前目标
            {
                print(YELLOW, "\none done");
                // 再试一次
                send_byte = -1;
                // 两次尝试后放弃抓取当前目标
                if (++max_attempt > 1)
                {
                    print(BOLDCYAN, "MAIN: tried for 2 times, grasping done");
                    land = false;
                    grasping_done = true;
                    max_attempt = 0;
                }
            }
            else
            {
                // 发送marker相对于目标的坐标
                std::string send_array = "#";
                send_array = send_array + std::to_string(-(marker_info.center.x / vis_size.width * 100 - target_info[1])) + "," + 
                            std::to_string(-(marker_info.center.y / vis_size.height * 100 - target_info[2])) + "," + 
                            std::to_string(curr_depth) + 
                            "\n";
                uart.send(send_array);
                // print(BOLDGREEN, "[marker relative position] " << send_array);
                std::cout << "\r[marker relative position] " << std::setw(6) << std::setfill(' ') << std::setprecision(3) << -(marker_info.center.x / vis_size.width * 100 - target_info[1]) << ", " 
                << std::setw(6) << std::setfill(' ') << std::setprecision(3) << -(marker_info.center.y / vis_size.height * 100 - target_info[2]) << ", " 
                <<  std::setw(6) << std::setfill(' ') << std::setprecision(3) << curr_depth;
                std::cout << "  [";
                for (int t=0; t<30; t++)
                {
                    if (t < (time(nullptr) - t_send)*30/TIME_PER_GRAP)
                    {
                        std::cout << '>';
                    }else
                    {
                        std::cout << ' ';
                    }
                }
                std::cout << ']' << ' ' << time(nullptr) - t_send << "s   ";
            }
        }
        int key = cv::waitKey(1) & 0xFF;
        key_1 = key;
        if (key == 255)
        {
            key = -1;
        }
        
        if (key != -1)
            rov_key = key;
        parse_key(key, quit, reset_id, conf_thresh, FLAGS_K, FLAGS_R, filter);
        if (save_a_count)
        {
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
    capture.receive_stop();
    print(BOLDGREEN, "bye!");
    return 0;
}
