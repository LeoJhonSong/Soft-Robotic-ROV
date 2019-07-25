//
// Created by sean on 7/11/19.
//
#include <utils.h>
#include <detector.h>
#include <uart.h>
#include <ruas.h>

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

DEFINE_int32(K, 100, "turbulence intensity. The greater, the intensive");
DEFINE_int32(R, 40, "Signal to Noise Ratio. The greater, the more serious of noise");
DEFINE_uint32(RUAS, 0, "0: skip; 1: clahe; 2: wiener+clahe");
DEFINE_uint32(NET_PHASE, 0, "0: skip; 1: netG; 2: netG+RefineDet; 3: RefineDet" );
DEFINE_uint32(SSD_DIM, 320, "" );
DEFINE_int32(MODE, -1, "-1: load video; >0 load camera" );
//DEFINE_string(SSD_MODEL, "../models/SSD320.pt", "" );


int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // load models
    torch::NoGradGuard no_grad_guard;
    std::string model_path;

    if(FLAGS_NET_PHASE==1) model_path = "./models/netG.pt";
    else if(FLAGS_NET_PHASE==2) model_path = "./models/G_SSD_320.pt";
    else if(FLAGS_SSD_DIM==512) model_path = "./models/SSD512.pt";
    else model_path = "./models/SSD320.pt";

    std::shared_ptr<torch::jit::script::Module> net = torch::jit::load(model_path);
    net->to(at::kCUDA);
    // load detector
    unsigned int num_classes = 5;
    int top_k = 200;
    float nms_thresh = 0.3;
    unsigned int tub = 5;
    std::vector<float> conf_thresh = {0.3, 0.3, 0.3, 0.3};
    float tub_thresh = 0.9;
    bool reset_id = false;
    Detector Detect(num_classes, top_k, nms_thresh, tub, FLAGS_SSD_DIM);

    // load filter
    CFilt filter(FLAGS_SSD_DIM, FLAGS_SSD_DIM, 3);
    if(FLAGS_RUAS>1) {
        filter.get_wf(FLAGS_K, FLAGS_R);
    }

    // load video
    cv::VideoCapture capture;
    if(FLAGS_MODE < 0) capture.open("/home/sean/data/UWdevkit/snippets/2.MP4");
    else capture.open(FLAGS_MODE);

    capture.set(CV_CAP_PROP_POS_FRAMES, 200);
    std::vector<int> vis_size(640, 480);

    // intermediate variable
    cv::Mat frame, img_float, img_vis;
    torch::Tensor img_tensor, fake_B, loc, conf, ota_feature, detections;
    std::vector<torch::jit::IValue> net_input, net_output;
    cv::cuda::GpuMat img_gpu;
    unsigned int loc_idex;

    while(capture.isOpened()){
        clock_t t1 = clock();
        capture.read(frame);
        cv::resize(frame, frame, cv::Size(FLAGS_SSD_DIM, FLAGS_SSD_DIM));
        if(FLAGS_RUAS == 1){
            filter.clahe_gpu(frame);
        }else if(FLAGS_RUAS == 2){
            filter.wiener_gpu(frame);
        }
//        cv::imshow("test", frame);
//        cv::waitKey(1);
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

        if(FLAGS_NET_PHASE>0) {
            if (FLAGS_NET_PHASE < 3 ) {
                cv::normalize(frame, img_float, -1, 1, cv::NORM_MINMAX, CV_32F);
            } else if (FLAGS_NET_PHASE == 3) {
                frame.convertTo(img_float, CV_32F);
                img_float = img_float - 128.0;
            }
            img_tensor = torch::from_blob(img_float.data, {1, FLAGS_SSD_DIM, FLAGS_SSD_DIM, 3}).to(torch::kCUDA);
            img_tensor = img_tensor.permute({0, 3, 1, 2});
            net_input.push_back(img_tensor);
            if(FLAGS_NET_PHASE == 1) {
                fake_B = net->forward(net_input).toTensor();
                loc_idex = 1;
                cudaDeviceSynchronize();
            }else if(FLAGS_NET_PHASE > 1) {
                net_output = net->forward(net_input).toTuple()->elements();
                cudaDeviceSynchronize();
                if (FLAGS_NET_PHASE == 2) {
                    fake_B = net_output.at(0).toTensor();
                    loc_idex = 1;
                } else if (FLAGS_NET_PHASE == 3) loc_idex = 0;
                loc = net_output.at(loc_idex).toTensor().to(torch::kCPU);
                conf = net_output.at(loc_idex + 1).toTensor().to(torch::kCPU);
                ota_feature = net_output.at(loc_idex + 2).toTensor().to(torch::kCPU);
            }
            net_input.pop_back();
            detections = Detect.detect(loc, conf, conf_thresh);
        }
        if(loc_idex == 1) img_vis = tensor2im(fake_B, vis_size);
        else {
            cv::cvtColor(frame, img_vis, cv::COLOR_BGR2RGB);
            cv::resize(img_vis, img_vis, cv::Size(vis_size[0], vis_size[1]));
        }
        Detect.visualization(img_vis, detections);
        clock_t t7 = clock();
        std::cout << "total: " << (t7 - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << std::endl;
    }

    return 0;
}