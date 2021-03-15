// for Linux file system
#include <dirent.h>
#include <sys/stat.h>
// for command line args
// for cuda and torch
#include <cuda_runtime.h>
#include <torch/script.h>
#include <torch/torch.h>
// for OpenCV
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
// for project
#include "color.h"
#include "detector.h"
#include "marker_detector.h"
#include "parallel_camera.h"
#include "ruas.h"
#include "visual_server.h"

const char *keys =
    "{k         | 100                   | turbulence intensity. The greater, the intensive}"
    "{r         | 50                    | Signal to Noise Ratio. The greater, the more serious of noise}"
    "{ruas      | 0                     | RUAS algorithm selection. 0: skip; 1: clahe; 2: wiener+clahe}"
    "{netg      | 256                   | netG dimension}"
    "{ssd       | 320                   | SSD dimension}"
    // "{tub       | true                  | }"
    "{mode      | 2                     | refinedet selection. 0: skip; 1: netG; 2: netG+RefineDet; 3: RefineDet}"
    "{stream    | file                  | source of video stream. file; link; camera}"
    "{cid       | 0                     | camera id if video stream come from camera}"
    "{address   | ./test/test.mp4       | address of video stream if not come from camera}"
    "{track     | true                  | track single target}"
    "{record    | false                 | record raw and processed video}"
    "{help      |                       | show help message}";

// 一些配置项
bool detect_scallop = true;
cv::Size vis_size(640, 360);
const int MARKER_OFFSET_X = 50;
const int MARKER_OFFSET_Y = 75;
unsigned int num_classes = 5; // 背景, 海参, 海胆, 扇贝, 海星
int top_k = 200;
float nms_thresh = 0.3;
// 背景, 海参, 海胆, 扇贝, 海星. 顺序由模型决定, 顺序决定了哪个类别优先级更高
std::vector<float> conf_thresh = {0, 0.6, 0.8, 0.3, 1.5};
float tub_thresh = 0.3;

// 在多个线程共享的全局变量
std::queue<cv::Mat> frame_queue, det_frame_queue;
detector::Visual_info visual_info;
bool threads_quit_flag;


cv::Mat tensor2im(torch::Tensor tensor)
{
    tensor = tensor[0].add(1.0).div(2.0).mul(255.0).permute({1, 2, 0}).to(torch::kU8).to(torch::kCPU);
    cv::Mat img(tensor.size(0), tensor.size(1), CV_8UC3);
    std::memcpy((void *)img.data, tensor.data_ptr(), sizeof(torch::kU8) * tensor.numel());
    return img;
}

void video_write(bool video_record_flag, std::string save_path)
{
    // 如果不录制视频, 退出视频录制线程
    if (!video_record_flag)
        return;
    char EXT[] = "MJPG";
    int ex1 = EXT[0] | (EXT[1] << 8) | (EXT[2] << 16) | (EXT[3] << 24);
    while (frame_queue.empty() || det_frame_queue.empty())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    // raw video
    cv::VideoWriter writer_raw;
    writer_raw.open("./record/" + save_path + "/" + save_path + "_raw.avi", ex1, 20, frame_queue.front().size(), true);
    if (!writer_raw.isOpened())
    {
        print(BOLDRED, "[ERROR] Can not open the raw output video");
    }
    // det video
    cv::VideoWriter writer_det;
    writer_det.open("./record/" + save_path + "/" + save_path + "_processed.avi", ex1, 20,
                    det_frame_queue.front().size(), true);
    if (!writer_det.isOpened())
    {
        print(BOLDRED, "[ERROR] Can not open the processed output video");
    }
    print(BOLDCYAN, "[Recorder] start");
    while (!threads_quit_flag)
    {
        if (!frame_queue.empty())
        {
            writer_raw << frame_queue.front();
            frame_queue.pop();
        }
        if (!det_frame_queue.empty())
        {
            writer_det << det_frame_queue.front();
            det_frame_queue.pop();
        }
    }
    print(BOLDCYAN, "[Recorder] quit");
    writer_raw.release();
    writer_det.release();
}

int main(int argc, char *argv[])
{
    print(GREEN, "loading, please wait...");
    // 读入命令行参数
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    int FLAGS_K = parser.get<int>("k");
    int FLAGS_R = parser.get<int>("r");
    int FLAGS_RUAS = parser.get<int>("ruas");
    int FLAGS_NETG_DIM = parser.get<int>("netg");
    int FLAGS_SSD_DIM = parser.get<int>("ssd");
    // bool FLAGS_TUB = parser.get<bool>("tub");
    bool FLAGS_TUB = true;
    int FLAGS_NET_PHASE = parser.get<int>("mode");
    cv::String FLAGS_STREAM = parser.get<cv::String>("stream");
    int FLAGS_CAMERA_ID = parser.get<int>("cid");
    cv::String FLAGS_ADDRESS;
    if (parser.has("address"))
    {
        FLAGS_ADDRESS = parser.get<cv::String>("address");
    }
    bool FLAGS_TRACK = parser.get<bool>("track");
    bool FLAGS_RECORD = parser.get<bool>("record");
    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }
    // load models
    torch::NoGradGuard no_grad_guard;
    std::string model_path;
    if (FLAGS_NET_PHASE == 1)
        model_path = "./models/netG.pt";
    else if (FLAGS_NET_PHASE == 2)
        model_path = "./models/Unet256_SSD320_wof.pt";
    else if (FLAGS_SSD_DIM == 512)
        model_path = "./models/SSD512_wof.pt";
    else
        model_path = "./models/SSD320_wof.pt";

    std::shared_ptr<torch::jit::script::Module> net = torch::jit::load(model_path);
    net->to(at::kCUDA);

    // load detector
    bool reset_id = false;
    detector::Detector detector(num_classes, top_k, nms_thresh, FLAGS_TUB, FLAGS_SSD_DIM, FLAGS_TRACK);

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
            if (FLAGS_STREAM == "file")
            {
                capture.open(FLAGS_ADDRESS);
                // 设置从视频数帧之后开始读, 跳过前戏
                capture.set(cv::CAP_PROP_POS_FRAMES, 1100);
            }
            else if (FLAGS_STREAM == "link")
                // capture.open("rtsp://admin:zhifan518@192.168.1.88/11");
                capture.open(FLAGS_ADDRESS);
            else if (FLAGS_STREAM == "camera")
                capture.open(FLAGS_CAMERA_ID);
            else
            {
                print(BOLDRED, "[ERROR] No such video stream type");
            }
        }
        catch (const char *msg)
        {
            print(RED, "[WARN] cannot open video");
            continue;
        }
    }

    // 视频录制进程设置
    bool video_record_flag = FLAGS_RECORD;
    // make record dir and file
    std::time_t t = std::time(nullptr);
    std::tm tm = *std::localtime(&t);
    std::locale locale_sys = std::locale("");
    std::stringstream save_path_ss;
    save_path_ss.imbue(locale_sys);
    save_path_ss << std::put_time(&tm, "%c");
    std::string save_path = save_path_ss.str();
    if (FLAGS_RECORD)
    {
        if (nullptr == opendir(("./record/" + save_path).c_str()))
            mkdir(("./record/" + save_path).c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
    }
    threads_quit_flag = false;
    // Start video recorder thread
    std::thread video_recorder(video_write, video_record_flag, save_path);
    // FIXME: 需要调试一下看看网络摄像头视频流延迟能否减小
    // start video receiver thread
    capture.receive_start();
    // Start visual info server thread
    std::thread visual_info_server(server::server_start);

    // 一些中间变量
    std::vector<float> target_loc = {0, 0, 0, 0};
    cv::Mat frame, img_float, img_vis;
    std::vector<torch::jit::IValue> net_input, net_output;
    torch::Tensor img_tensor, fake_B, loc, conf;
    unsigned char loc_index = 0;

    // marker detector
    // 初始化的size要对应上后面输入图片的size,看到时候用哪个图片(原始的frame, net_G输出的fake_B,
    // 或者resize后的img_vis)比较好 marker::MarkerDetector marker_detector(frame.size());
    marker::MarkerDetector marker_detector(vis_size);
    marker::MarkerInfo marker_info_current;
    marker::MarkerInfo marker_info;

    while (capture.isOpened())
    {
        // 获取视频流中最新帧
        bool read_ret = capture.read(frame);
        if ((cv::waitKey(1) & 0xFF) == 27) // ESC
            threads_quit_flag = true;
        if (!read_ret || threads_quit_flag)
            break;
        // pre processing
        cv::resize(frame, frame, cv::Size(FLAGS_NETG_DIM, FLAGS_NETG_DIM));
        if (FLAGS_RUAS == 1)
        {
            filter.clahe_gpu(frame);
        }
        else if (FLAGS_RUAS == 2)
        {
            filter.wiener_gpu(frame);
        }
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        // run net
        if (FLAGS_NET_PHASE != 3) // when not using netG as detect only
        {
            cv::normalize(frame, img_float, -1, 1, cv::NORM_MINMAX, CV_32F);
        }
        else
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
            loc_index = 1;
            cudaDeviceSynchronize();
        }
        else if (FLAGS_NET_PHASE > 1)
        {
            net_output = net->forward(net_input).toTuple()->elements();
            cudaDeviceSynchronize();
            if (FLAGS_NET_PHASE == 2)
            {
                fake_B = net_output.at(0).toTensor();
                loc_index = 1;
            }
            else if (FLAGS_NET_PHASE == 3)
                loc_index = 0;
            loc = net_output.at(loc_index).toTensor().to(torch::kCPU);
            conf = net_output.at(loc_index + 1).toTensor().to(torch::kCPU);
        }
        if (loc_index == 1)
            img_vis = tensor2im(fake_B);
        else
            img_vis = frame;
        net_input.pop_back();
        // detect
        cv::cvtColor(img_vis, img_vis, cv::COLOR_BGR2RGB);
        cv::resize(img_vis, img_vis, vis_size);

        // detect marker
        // TODO: if (visual_info.arm_is_working)
        marker_info_current =
            marker_detector.detect_single_marker(img_vis, true, marker::VER_OPENCV, marker::MODE_DETECT);
        marker_info = marker_info_current;
        if (marker_info_current.center.x > 0 && marker_info_current.center.y > 0)
        {
            // 补偿偏置
            marker_info.center.x += MARKER_OFFSET_X;
            marker_info.center.y += MARKER_OFFSET_Y;
            // update visual_info.marker*
            visual_info.has_marker = true;
            visual_info.marker_position =
                cv::Point2f(marker_info.center.x / vis_size.width, marker_info.center.y / vis_size.height);
        }
        else
        {
            visual_info.has_marker = false;
            visual_info.marker_position = cv::Point2f(0, 0);
        }
        // 补偿后原点
        cv::circle(img_vis, marker_info.center, 6, cv::Scalar(0, 0, 255), -1, 8, 0);
        // print(BOLDYELLOW, "x: " << marker_info.center.x << " y: " << marker_info.center.y);

        target_loc = detector.detect_and_visualize(loc, conf, conf_thresh, tub_thresh, reset_id, detect_scallop,
                                                   img_vis); // cx, cy, width, height
        if (video_record_flag && !threads_quit_flag)
            det_frame_queue.push(img_vis);
        // update visual_info.target*
        if (detector.track_id > -1)
        {
            visual_info.has_target = true;
            visual_info.target_class = detector.tracking_class;
            visual_info.target_id = detector.track_id;
            visual_info.target_center = cv::Point2f(target_loc[0], target_loc[1]);
            visual_info.target_shape = cv::Point2f(target_loc[2], target_loc[3]);
        }
        else // clear visual_info.target*
        {
            visual_info.has_target = false;
            visual_info.target_class = 0;
            visual_info.target_id = -1;
            visual_info.target_center = cv::Point2f(0, 0);
            visual_info.target_shape = cv::Point2f(0, 0);
        }
        // print(GREEN, "[Debug] " << visual_info.target_center);
    }
    print(BOLDGREEN, "[Info] holothurian: " << detector.get_class_num(1) << ", echinus: " << detector.get_class_num(2)
                                            << ", scallop: " << detector.get_class_num(3));
    // wait for child threads to quit
    video_recorder.join();
    capture.receive_stop();
    visual_info_server.join();

    print(BOLDYELLOW, "bye!");
    return 0;
}
