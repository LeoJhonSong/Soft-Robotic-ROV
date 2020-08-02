//
// Created by sean on 7/11/19.
//

#include "utils.h"
#include "color.h"
#include <chrono>

cv::Mat tensor2im(torch::Tensor tensor) {
    tensor = tensor[0].add(1.0).div(2.0).mul(255.0).permute({1,2,0}).to(torch::kU8).to(torch::kCPU);
    cv::Mat img(tensor.size(0), tensor.size(1), CV_8UC3);
    std::memcpy((void*)img.data, tensor.data_ptr(), sizeof(torch::kU8)*tensor.numel());
    return img;
}

void clip(float& n, float lower, float upper) {
    n = std::max(lower, std::min(n, upper));
}

void clip(int& n, int lower, int upper) {
    n = std::max(lower, std::min(n, upper));
}

extern bool rov_half_speed, manual_stop, save_a_frame, save_a_count;
extern float half_scale, adjust_scale;
void parse_key(int key, bool& quit, bool& reset_id, std::vector<float>& conf_thresh, int& FLAGS_K, int& FLAGS_R, CFilt& filter){
    switch (key){
        case 27:  // esc
            quit = true;
            break;
        // case 50:  // 2
        //     conf_thresh.at(0) += 0.1;
        //     clip(conf_thresh.at(0), 0.001, 1.0);
        //     print(YELLOW, "KEY: conf_thresh: " << conf_thresh);
        //     break;
        // case 49:  // 1
        //     conf_thresh.at(0) -= 0.1;
        //     clip(conf_thresh.at(0), 0.001, 1.0);
        //     print(YELLOW, "KEY: conf_thresh: " << conf_thresh);
        //     break;
        // case 119:  // w
        //     conf_thresh.at(1) += 0.1;
        //     clip(conf_thresh.at(1), 0.001, 1.0);
        //     print(YELLOW, "KEY: conf_thresh: " << conf_thresh);
        //     break;
        // case 113:  // q
        //     conf_thresh.at(1) -= 0.1;
        //     clip(conf_thresh.at(1), 0.001, 1.0);
        //     print(YELLOW, "KEY: conf_thresh: " << conf_thresh);
        //     break;
        // case 115:  // s
        //     conf_thresh.at(2) += 0.1;
        //     clip(conf_thresh.at(2), 0.001, 1.0);
        //     print(YELLOW, "KEY: conf_thresh: " << conf_thresh);
        //     break;
        // case 97:  // a
        //     conf_thresh.at(2) -= 0.1;
        //     clip(conf_thresh.at(2), 0.001, 1.0);
        //     print(YELLOW, "KEY: conf_thresh: " << conf_thresh);
        //     break;
        // case 120:  // x
        //     conf_thresh.at(3) += 0.1;
        //     clip(conf_thresh.at(3), 0.001, 1.0);
        //     print(YELLOW, "KEY: conf_thresh: " << conf_thresh);
        //     break;
        // case 122:  // z
        //     conf_thresh.at(3) -= 0.1;
        //     clip(conf_thresh.at(3), 0.001, 1.0);
        //     print(YELLOW, "KEY: conf_thresh: " << conf_thresh);
        //     break;
        case 101: // e
            save_a_frame = true;
            break;
        case 118: // v
            save_a_count = true;
            break;
        // case 57: // 9
        //     half_scale -= 0.1;
        //     clip(half_scale, 0.1, 10.0);
        //     print(YELLOW, "KEY: half_scale: " << half_scale);
        //     break;
        // case 48: // 0
        //     half_scale += 0.1;
        //     clip(half_scale, 0.1, 10.0);
        //     print(YELLOW, "KEY: half_scale: " << half_scale);
        //     break;
        // case 55: // 7
        //     adjust_scale -= 0.02;
        //     clip(adjust_scale, 0.1, 10.0);
        //     print(YELLOW, "KEY: adjust_scale: " << adjust_scale);
        //     break;
        // case 56: // 8
        //     adjust_scale += 0.02;
        //     clip(adjust_scale, 0.1, 10.0);
        //     print(YELLOW, "KEY: adjust_scale: " << adjust_scale);
        //     break;
        case 75:  // K (shift + k)
            FLAGS_K += 10;
            clip(FLAGS_K, 0, 300);
            filter.get_wf(FLAGS_K, FLAGS_R);
            std::cout << "main: K: " << FLAGS_K << std::endl;
            break;
        case 107:  // k
            FLAGS_K -= 10;
            clip(FLAGS_K, 0, 300);
            filter.get_wf(FLAGS_K, FLAGS_R);
            std::cout << "main: K: " << FLAGS_K << std::endl;
            break;
        case 114:  // r
            reset_id = true;
            break;
        case 109:  // m
            if (rov_half_speed) {
                rov_half_speed = false;
                print(YELLOW, "KEY: rov full speed");
            }
            else if (!rov_half_speed) {
                rov_half_speed = true;
                print(YELLOW, "KEY: rov half speed");
            }
            break;
        case 98:  // b
            if (!manual_stop) {
                print(YELLOW, "KEY: manually stop a rov behavior and init its states");
                init_state();
            }else{
                print(YELLOW, "KEY: manual_stop = false");
                manual_stop = false;
            }
            break;
        default:
            break;
    }
}

extern std::queue<cv::Mat> frame_queue, det_frame_queue;
extern std::queue<std::pair<cv::Mat, unsigned int>> img_queue;
extern int frame_w, frame_h, ex1, send_byte, rov_key;
extern bool video_write_flag, grasping_done, land, second_dive;
extern std::string save_path;
extern unsigned char max_attempt;
extern cv::Size vis_size;
void video_write(){
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

void init_state(){
    rov_key = 32;
    land = false;
    send_byte = -1;
    max_attempt = 0;
    grasping_done = false;
    second_dive = false;
    // manual_stop = true;
    // delay(1);
    manual_stop = true;
}

void delay(int s)
{
    time_t now = time(nullptr);
    while (time(nullptr) - now < s);
}

void delay_ms(int ms){
    std::chrono::milliseconds now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
    while (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() - now.count() < ms);
}