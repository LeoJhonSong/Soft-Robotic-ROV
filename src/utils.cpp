//
// Created by sean on 7/11/19.
//

#include <utils.h>


cv::Mat tensor2im(torch::Tensor tensor, std::vector<int> vis_size) {
    tensor = torch::upsample_bilinear2d(tensor, {vis_size.at(1), vis_size.at(0)}, true);
//    tensor = tensor[0].add(128.0).permute({1,2,0}).to(torch::kU8).to(torch::kCPU);
    tensor = tensor[0].add(1.0).div(2.0).mul(255.0).permute({1,2,0}).to(torch::kU8).to(torch::kCPU);
    cv::Mat img(tensor.size(0), tensor.size(1), CV_8UC3);
    std::memcpy((void*)img.data, tensor.data_ptr(), sizeof(torch::kU8)*tensor.numel());
    cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
//    cv::resize(img, img, vis_size);
    return img;
}

int clip(int n, int lower, int upper) {
    return std::max(lower, std::min(n, upper));
}

void clip(float& n, float lower, float upper) {
    n = std::max(lower, std::min(n, upper));
}

void parse_key(int key, bool& quit, bool& reset_id, std::vector<float>& conf_thresh){
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
            clip(conf_thresh.at(0), 0.0, 1.0);
            std::cout << "conf_thresh: " << conf_thresh << std::endl;
            break;
        }
        case 49: {  // 1
            conf_thresh.at(0) -= 0.1;
            clip(conf_thresh.at(0), 0.0, 1.0);
            std::cout << "conf_thresh: " << conf_thresh << std::endl;
            break;
        }
        case 87:{}
        case 119: {  // w
            conf_thresh.at(1) += 0.1;
            clip(conf_thresh.at(1), 0.0, 1.0);
            std::cout << "conf_thresh: " << conf_thresh << std::endl;
            break;
        }
        case 81:{}
        case 113: {  // q
            conf_thresh.at(1) -= 0.1;
            clip(conf_thresh.at(1), 0.0, 1.0);
            std::cout << "conf_thresh: " << conf_thresh << std::endl;
            break;
        }
        case 83:{}
        case 115: {  // s
            conf_thresh.at(2) += 0.1;
            clip(conf_thresh.at(2), 0.0, 1.0);
            std::cout << "conf_thresh: " << conf_thresh << std::endl;
            break;
        }
        case 65:{}
        case 97: {  // a
            conf_thresh.at(2) -= 0.1;
            clip(conf_thresh.at(2), 0.0, 1.0);
            std::cout << "conf_thresh: " << conf_thresh << std::endl;
            break;
        }
        case 88:
        case 120: {  // x
            conf_thresh.at(3) += 0.1;
            clip(conf_thresh.at(3), 0.0, 1.0);
            std::cout << "conf_thresh: " << conf_thresh << std::endl;
            break;
        }
        case 90:
        case 122: {  // z
            conf_thresh.at(3) -= 0.1;
            clip(conf_thresh.at(3), 0.0, 1.0);
            std::cout << "conf_thresh: " << conf_thresh << std::endl;
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