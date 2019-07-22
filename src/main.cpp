//
// Created by sean on 7/11/19.
//
#include <utils.h>
#include <detector.h>
#include <uart.h>

#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <memory>
#include <string>
#include <vector>


int main(int argc, const char* argv[]) {
    // load models
    // torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    torch::NoGradGuard no_grad_guard;
    std::shared_ptr<torch::jit::script::Module> netG = torch::jit::load("../models/netG.pt");
    netG->to(at::kCUDA);
    std::shared_ptr<torch::jit::script::Module> SSD = torch::jit::load("../models/SSD320.pt");
    SSD->to(at::kCUDA);

    // load detector
    unsigned int num_classes = 5;
    int top_k = 200;
    float nms_thresh = 0.3;
    unsigned int tub = 5;
    int ssd_dim = 320;
    std::vector<float> conf_thresh = {0.3, 0.3, 0.3, 0.3};
    float tub_thresh = 0.9;
    bool reset_id = false;
    Detector Detect(num_classes, top_k, nms_thresh, tub, ssd_dim);

    // load video
    cv::VideoCapture capture("../2.MP4");
    capture.set(cv::CAP_PROP_POS_FRAMES, 100);

    // intermediate variable
    cv::Mat frame, img_float, img_vis;
    torch::Tensor fake_B, loc, conf, ota_feature, detections;
    std::vector<torch::jit::IValue> netG_input, SSD_input, SSD_output;

    while(capture.isOpened()){
        clock_t t1 = clock();
        capture.read(frame);
        cv::resize(frame, frame, cv::Size(ssd_dim, ssd_dim));
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        cv::normalize(frame, img_float, -1, 1, cv::NORM_MINMAX, CV_32F);
        clock_t t1_1 = clock();
        auto img_tensor = torch::from_blob(img_float.data, {1, ssd_dim, ssd_dim, 3}).to(torch::kCUDA);
        img_tensor = img_tensor.permute({0, 3, 1, 2});
        clock_t t2 = clock();
        netG_input.push_back(img_tensor);
        fake_B = netG->forward(netG_input).toTensor();
        netG_input.pop_back();
        clock_t t3 = clock();
        SSD_input.push_back(fake_B.add(1.0).div(2.0).mul(255.0).sub(128.));
        SSD_output = SSD->forward(SSD_input).toTuple()->elements();
        SSD_input.pop_back();
        clock_t t4 = clock();
        loc = SSD_output.at(0).toTensor().to(torch::kCPU);
        conf = SSD_output.at(1).toTensor().to(torch::kCPU);
        clock_t t5 = clock();
//        ota_feature = SSD_output.at(2).toTensor().to(torch::kCPU);
        detections = Detect.detect(loc, conf, conf_thresh);
        clock_t t6 = clock();
        img_vis = tensor2im(fake_B);
        Detect.visualization(img_vis, detections);
        clock_t t7 = clock();
        std::cout << "total: " << (t7 - t1) * 1.0 / CLOCKS_PER_SEC * 1000
                  <<  ", pre: " << (t1_1 - t1) * 1.0 / CLOCKS_PER_SEC * 1000
                  <<  ", cpu2gpu: " << (t2 - t1_1) * 1.0 / CLOCKS_PER_SEC * 1000
                  <<  ", netG: " << (t3 - t2) * 1.0 / CLOCKS_PER_SEC * 1000
                  <<  ", SSD: "  << (t4 - t3) * 1.0 / CLOCKS_PER_SEC * 1000
                  <<  ", gpu2cpu: "  << (t5 - t4) * 1.0 / CLOCKS_PER_SEC * 1000
                  <<  ", detection: "  << (t6 - t5) * 1.0 / CLOCKS_PER_SEC * 1000
                  <<  ", vis: "  << (t7 - t6) * 1.0 / CLOCKS_PER_SEC * 1000 << std::endl;

//        cv::imshow("test", img_vis);
//        cv::waitKey(1);
    }
}