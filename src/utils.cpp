//
// Created by sean on 7/11/19.
//

#include <utils.h>


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

void parse_key(int key, bool& quit, bool& reset_id, std::vector<float>& conf_thresh, int& FLAGS_K, int& FLAGS_R, CFilt& filter){
    switch (key){
        case 32:{  // space
            while(true) {
                int key_c = cv::waitKey(1);
                if(key_c == 32) break;
            }
            break;
        }
        case 27:{  // esc
            quit = true;
            break;
        }
        case 50: {  // 2
            conf_thresh.at(0) += 0.1;
            clip(conf_thresh.at(0), 0.001, 1.0);
            std::cout << "conf_thresh: " << conf_thresh << std::endl;
            break;
        }
        case 49: {  // 1
            conf_thresh.at(0) -= 0.1;
            clip(conf_thresh.at(0), 0.001, 1.0);
            std::cout << "conf_thresh: " << conf_thresh << std::endl;
            break;
        }
        case 119: {  // w
            conf_thresh.at(1) += 0.1;
            clip(conf_thresh.at(1), 0.001, 1.0);
            std::cout << "conf_thresh: " << conf_thresh << std::endl;
            break;
        }
        case 113: {  // q
            conf_thresh.at(1) -= 0.1;
            clip(conf_thresh.at(1), 0.001, 1.0);
            std::cout << "conf_thresh: " << conf_thresh << std::endl;
            break;
        }
        case 115: {  // s
            conf_thresh.at(2) += 0.1;
            clip(conf_thresh.at(2), 0.001, 1.0);
            std::cout << "conf_thresh: " << conf_thresh << std::endl;
            break;
        }
        case 97: {  // a
            conf_thresh.at(2) -= 0.1;
            clip(conf_thresh.at(2), 0.001, 1.0);
            std::cout << "conf_thresh: " << conf_thresh << std::endl;
            break;
        }
        case 120: {  // x
            conf_thresh.at(3) += 0.1;
            clip(conf_thresh.at(3), 0.001, 1.0);
            std::cout << "conf_thresh: " << conf_thresh << std::endl;
            break;
        }
        case 122: {  // z
            conf_thresh.at(3) -= 0.1;
            clip(conf_thresh.at(3), 0.001, 1.0);
            std::cout << "conf_thresh: " << conf_thresh << std::endl;
            break;
        }
        case 107: {  // k
            FLAGS_K += 10;
            clip(FLAGS_K, 0, 300);
            filter.get_wf(FLAGS_K, FLAGS_R);
            std::cout << "K: " << FLAGS_K << std::endl;
            break;
        }
        case 108: {  // k
            FLAGS_K -= 10;
            clip(FLAGS_K, 0, 300);
            filter.get_wf(FLAGS_K, FLAGS_R);
            std::cout << "K: " << FLAGS_K << std::endl;
            break;
        }
        case 114: {  // r
            reset_id = true;
            break;
        }
    }
//    elif key == 50: #2
//    conf_list[0] = np.around(np.clip(conf_list[0] + 0.1, 0., 1.), 1)
//    elif key == 49: #1
//    conf_list[0] = np.around(np.clip(conf_list[0] - 0.1, 0., 1.), 1)
//    elif key in [87, 119]: #w:
//    conf_list[1] = np.around(np.clip(conf_list[1] + 0.1, 0., 1.), 1)
//    elif key in [81, 113]: #q
//            conf_list[1] = np.around(np.clip(conf_list[1] - 0.1, 0., 1.), 1)
//    elif key in [83, 115]: #s
//            conf_list[2] = np.around(np.clip(conf_list[2] + 0.1, 0., 1.), 1)
//    elif key in [65, 97]: #a
//            conf_list[2] = np.around(np.clip(conf_list[2] - 0.1, 0., 1.), 1)
//    elif key in [88, 120]: #x:
//    conf_list[3] = np.around(np.clip(conf_list[3] + 0.1, 0., 1.), 1)
//    elif key in [90, 122]: #z:
//    conf_list[3] = np.around(np.clip(conf_list[3] - 0.1, 0., 1.), 1)
//    elif key == 46:  # .
//    tub_thresh = np.around(np.clip(tub_thresh + 0.1, 0., 1.5), 1)
//    elif key == 44:  # ,
//    tub_thresh = np.around(np.clip(tub_thresh - 0.1, 0., 1.5), 1)
//    elif key in [82, 114]:  # ,
//    max_id = [torch.tensor(0.),]*len(UW_CLASSES)
//    reset_id = True
}

extern std::string video_name;
extern int frame_w, frame_h, ex1;
extern bool raw_write_flag;
extern std::queue<cv::Mat> frame_queue;

void raw_write(){
    cv::VideoWriter writer_raw;
    writer_raw.open(video_name+"_raw.mp4", ex1, 20, cv::Size(frame_w, frame_h), true);
    if(!writer_raw.isOpened()){
        std::cout << "Can not open the output video for raw write" << std::endl;
    }
    while(raw_write_flag) {
        if (!frame_queue.empty()) {
            writer_raw << frame_queue.front();
            frame_queue.pop();
        }
    }
    std::cout << "raw_write thread qiut" << std::endl;
    writer_raw.release();
}