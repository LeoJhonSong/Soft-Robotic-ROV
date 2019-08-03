//
// Created by sean on 7/17/19.
//

#include "detector.h"

Detector::Detector(unsigned int num_classes, int top_k,
        float nms_thresh, unsigned char tub, int ssd_dim){
    this->num_classes = num_classes;
    this->top_k = top_k;
    this->nms_thresh = nms_thresh;
    this->tub = tub;
    this->ssd_dim = ssd_dim;
    if(this->tub > 0) {
        this->init_tubelets();
        this->hold_len = 10;
//        this->match_score = 0.6;
//        this->feature_size = 7;
//        this->output = torch::zeros({1, this->num_classes, this->top_k, 5}, torch::kFloat);
    }
    this->output = torch::zeros({1, this->num_classes, this->top_k, 6}, torch::kFloat);
//    bool uart_open_flag, uart_init_flag;
//    Uart uart_("ttyUSB0", 115200);
//    uart_open_flag = uart_.openFile();
//    if (!uart_open_flag)
//        std::cout << "UART fails to open " << std::endl;
//    uart_init_flag = uart_.initPort();
//    if (!uart_init_flag)
//        std::cout << "UART fails to be inited " << std::endl;
//    this->uart = uart_;
    this->log_params();
}

void Detector::log_params(){
    std::cout << "num_classes: " << num_classes << std::endl;
    std::cout << "top_k: " << top_k << std::endl;
    std::cout << "nms_thresh: " << nms_thresh << std::endl;
    std::cout << "tub: " << tub << std::endl;
    std::cout << "ssd_dim: " << ssd_dim << std::endl;
    std::cout << "out size: " << output.sizes() << output.dtype() << std::endl;
    std::cout<< "tubelets class size: " << this->tubelets.size() << std::endl;
}

void Detector::detect(const torch::Tensor& loc, const torch::Tensor& conf, std::vector<float> conf_thresh){
    this->output.zero_();
    for(unsigned int cl=1; cl<this->num_classes; cl++){
        torch::Tensor c_mask = conf[cl].gt(conf_thresh.at(cl-1));
        if(c_mask.sum().item<int >() == 0){
            continue;
        }
        torch::Tensor scores = conf[cl].masked_select(c_mask);
        torch::Tensor l_mask = c_mask.unsqueeze(1).expand_as(loc);
        torch::Tensor boxes = loc.masked_select(l_mask).view({-1, 4});
        std::tuple<torch::Tensor, int> nms_result = nms(boxes.mul(this->ssd_dim), scores);
        torch::Tensor keep = std::get<0>(nms_result);
        int count = std::get<1>(nms_result);

        torch::Tensor nms_score = scores.index_select(0, keep);
        torch::Tensor nms_box = boxes.index_select(0, keep);
        torch::Tensor identity = torch::ones({count}).fill_(-1);
        this->output[0][cl].slice(0, 0, count) = torch::cat({nms_score.unsqueeze(1), nms_box, identity.unsqueeze(1)}, 1);
    }
//    return this->output;
}

void Detector::detect(const torch::Tensor& loc, const torch::Tensor& conf, std::vector<float> conf_thresh, float tub_thresh, bool reset){
    this->output.zero_();
    for(unsigned char cl=1; cl<this->num_classes; cl++){
        torch::Tensor c_mask = conf[cl].gt(conf_thresh.at(cl-1));
        if(c_mask.sum().item<int >() == 0){
            this->delete_tubelets(cl);
            continue;
        }
        torch::Tensor scores = conf[cl].masked_select(c_mask);
        torch::Tensor l_mask = c_mask.unsqueeze(1).expand_as(loc);
        torch::Tensor boxes = loc.masked_select(l_mask).view({-1, 4});
        std::tuple<torch::Tensor, int> nms_result = nms(boxes.mul(this->ssd_dim), scores);
        torch::Tensor keep = std::get<0>(nms_result);
        int count = std::get<1>(nms_result);
        if(reset) this->init_tubelets();

        torch::Tensor nms_score = scores.index_select(0, keep);
        torch::Tensor nms_box = boxes.index_select(0, keep);
        torch::Tensor identity = torch::ones({count}).fill_(-1);
        if(!this->tubelets.at(cl).empty()){
            torch::Tensor iou = this->iou(nms_box, cl);
            std::tuple<torch::Tensor,torch::Tensor> max_info = torch::max(iou, 1);
            torch::Tensor max_simi = std::get<0>(max_info);
            torch::Tensor max_idx = std::get<1>(max_info);
            torch::Tensor matched_mask = max_simi.gt(tub_thresh);
            for(unsigned char mt=0; mt<count; mt++) {
                if (matched_mask[mt].item<int>() > 0)
                    identity[mt] = this->ides.at(cl).at(max_idx[mt].item<int>());
            }
        }
        torch::Tensor new_id_mask = identity.eq(-1);
        if(new_id_mask.sum().item<int32_t >()>0){
            int current = this->history_max_ides[cl].item<int32_t >() + 1;
            torch::Tensor new_id = torch::arange(current, current + new_id_mask.sum().item<int32_t >() );
            this->history_max_ides[cl] = new_id[new_id.size(0)-1];
            int nid = 0;
            for(unsigned char m=0; m<identity.size(0); m++)
                if(new_id_mask[m].item<int >() > 0) identity[m] = new_id[nid++];
        }
//        std::cout << identity << ", " << this->ides_set.at(cl) << std::endl;
        for(unsigned int tc=0; tc<count; tc++){
            int curr_id = identity[tc].item<int>();
            if(this->tubelets.at(cl).find(curr_id) == this->tubelets.at(cl).end()){
                this->tubelets.at(cl)[curr_id] = std::pair<torch::Tensor, int>{nms_box[tc].unsqueeze(0), this->hold_len+1};
            }else{
                this->ides_set.at(cl).erase(curr_id);
                torch::Tensor new_tube = torch::cat({nms_box[tc].unsqueeze(0), this->tubelets.at(cl)[curr_id].first}, 0);
                if(new_tube.size(0) > this->tub)
                    new_tube = new_tube.slice(0, 0, this->tub);
                this->tubelets.at(cl)[curr_id] = std::pair<torch::Tensor, int>{new_tube, this->hold_len+1};
                nms_box[tc] = new_tube.mean(0);
            }
        }
        this->output[0][cl].slice(0, 0, count) = torch::cat({nms_score.unsqueeze(1), nms_box, identity.unsqueeze(1)}, 1);
        int non_matched_size = 0;
        for(auto s:this->ides_set.at(cl) ) {
            torch::Tensor no_matched_box = this->tubelets.at(cl)[s].first[0];
            if (no_matched_box.lt(0.1).sum().item<uint8_t >()>0 || no_matched_box.gt(0.9).sum().item<uint8_t >()>0) continue;
            this->output[0][cl].slice(0, count + non_matched_size, count + non_matched_size+1) = torch::cat({torch::zeros({1}).fill_(0.01), this->tubelets.at(cl)[s].first[0], torch::zeros({1}).fill_(s)}, 0);
            non_matched_size++;
        }
//        std::cout << "after erase: "<< count << ", " << non_matched_size << std::endl;
        this->delete_tubelets(cl);
    }
//    return this->output;
}

std::tuple<torch::Tensor, int> Detector::nms(const torch::Tensor& boxes, const torch::Tensor& scores){
    torch::Tensor keep = torch::zeros(scores.sizes()).to(torch::kLong);
    int count = 0;
    std::tuple<torch::Tensor, int> nms_result(keep, count);
    if(boxes.numel() == 0){
        return nms_result;
    }
    torch::Tensor x1 = boxes.slice(1, 0, 1).squeeze(-1);
    torch::Tensor y1 = boxes.slice(1, 1, 2).squeeze(-1);
    torch::Tensor x2 = boxes.slice(1, 2, 3).squeeze(-1);
    torch::Tensor y2 = boxes.slice(1, 3, 4).squeeze(-1);
    torch::Tensor area = torch::mul(x2-x1, y2-y1);
    std::tuple<torch::Tensor, torch::Tensor> sorted_scores = scores.sort(0, false);
    torch::Tensor idx = std::get<1>(sorted_scores);
    if(idx.size(0) > this->top_k) idx = idx.slice(0, idx.size(0)-this->top_k, idx.size(0));
    while(idx.numel() > 0){
        int i = idx[idx.size(0)-1].item<int >();
        keep[count++] = i;
        if(idx.size(0) == 1) break;
        idx = idx.slice(0, 0, idx.size(0)-1);
        torch::Tensor inners = (x2.index_select(0, idx).clamp_max(x2[i].item<float>()) - x1.index_select(0, idx).clamp_min(x1[i].item<float>())).clamp_min(0.0)
                * (y2.index_select(0, idx).clamp_max(y2[i].item<float>()) - y1.index_select(0, idx).clamp_min(y1[i].item<float>())).clamp_min(0.0);;
        torch::Tensor rem_areas = area.index_select(0, idx);
        torch::Tensor unions = (rem_areas - inners) + area[i];
        torch::Tensor IoU = inners / unions;
        idx = idx.masked_select(IoU.le(this->nms_thresh));
    }
    std::get<0>(nms_result) = keep.slice(0, 0, count);
    std::get<1>(nms_result) = count;
    return nms_result;
}

torch::Tensor Detector::iou(const torch::Tensor& boxes, unsigned char cl){
    std::map<int, std::pair<torch::Tensor, int>> tubs = this->tubelets.at(cl);
    int tubs_size = tubs.size();
    torch::Tensor iou = torch::zeros({boxes.size(0), tubs_size});
    torch::Tensor x1 = boxes.slice(1, 0, 1).squeeze(-1);
    torch::Tensor y1 = boxes.slice(1, 1, 2).squeeze(-1);
    torch::Tensor x2 = boxes.slice(1, 2, 3).squeeze(-1);
    torch::Tensor y2 = boxes.slice(1, 3, 4).squeeze(-1);
    torch::Tensor area = torch::mul(x2-x1, y2-y1);
    unsigned char i = 0;
    for(auto& tube:tubs){
        this->ides.at(cl).push_back(tube.first);
        this->ides_set.at(cl).insert(tube.first);
        torch::Tensor last_tube = tube.second.first[0];
        torch::Tensor area_tube = torch::mul(last_tube[2] - last_tube[0], last_tube[3] - last_tube[1]);
        torch::Tensor inner = (x2.clamp_max(last_tube[2].item<float>())-x1.clamp_min(last_tube[0].item<float>())).clamp_min(0.0)
                * (y2.clamp_max(last_tube[3].item<float>())-y1.clamp_min(last_tube[1].item<float>())).clamp_min(0.0);
        torch::Tensor unions = area - inner + area_tube;
        iou.slice(1, i, i+1) = inner.div(unions).unsqueeze(-1);
        i++;
    }
//    std::cout << ", " << this->ides.at(cl).size() << ", "<<  this->ides_set.at(cl).size()  << std::endl;
    return iou;
}


void Detector::visualization(cv::Mat& img){
    for(unsigned char j=1; j<this->output.size(1); j++){
        torch::Tensor dets = this->output[0][j];
        if(dets.sum().item<float>() == 0) continue;
        torch::Tensor mask = dets.slice(1, 0, 1).gt(0.0).expand_as(dets);
//        std::cout << "vis: " << mask.sum() / 6 << std::endl;
        dets = dets.masked_select(mask);
        dets = dets.view({dets.size(0)/mask.size(1), mask.size(1)});
        torch::Tensor boxes = dets.slice(1, 1, 5);
        boxes.slice(1, 0, 1) *= img.cols;
        boxes.slice(1, 2, 3) *= img.cols;
        boxes.slice(1, 1, 2) *= img.rows;
        boxes.slice(1, 3, 4) *= img.rows;
        torch::Tensor scores = dets.slice(1, 0, 1).squeeze(1);
        torch::Tensor ids = dets.slice(1, 5, 6).squeeze(1);

        for(unsigned char i=0; i<boxes.size(0); i++){
            int x1 = boxes[i][0].item<int>();
            int y1 = boxes[i][1].item<int>();
            int x2 = boxes[i][2].item<int>();
            int y2 = boxes[i][3].item<int>();
            float ratio = (float)(x2-x1)*(float)(y2-y1)/(float)(img.cols*img.rows);
            if(ratio>0.1 || ratio<0.01) continue;
            cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), this->color.at(j-1), 2, 1, 0);
            if(ids[i].item<int>()>=0)
                cv::putText(img, std::to_string(ids[i].item<int>()), cv::Point(x1, y1-5), 1, 1, this->color.at(j-1));
        }
    }
    cv::imshow("ResDet", img);
}

void Detector::init_tubelets(){
    for(unsigned char i=0; i<num_classes; i++) {
        this->tubelets.emplace_back(std::map<int, std::pair<torch::Tensor, int>>{});
        this->ides.emplace_back(std::vector<int>{});
        this->tubelets.at(i).clear();
        this->ides.at(i).clear();
        this->ides_set.emplace_back(std::set<int>{});
        this->ides_set.at(i).clear();
    }
    this->history_max_ides = torch::ones({num_classes}).fill_(-1);
}

void Detector::delete_tubelets(unsigned char cl){
    std::vector<int> delet_list;
    std::map<int, std::pair<torch::Tensor, int>> tubs = this->tubelets.at(cl);
    for(auto& tube:tubs){
        if(--tube.second.second <= 0)
            delet_list.push_back(tube.first);
        this->tubelets.at(cl)[tube.first].second = tube.second.second;
    }
    for(auto id:delet_list)
        this->tubelets.at(cl).erase(id);
    this->ides.at(cl).clear();
    this->ides_set.at(cl).clear();
}

void Detector::uart_send(unsigned char cls, Uart& uart){
    torch::Tensor dets = this->output[0][cls][0];
    std::vector<char> send_list;
    send_list.push_back(127);
    send_list.push_back((char)(dets[1].item<float>()*100));
    send_list.push_back((char)(dets[2].item<float>()*100));
    send_list.push_back((char)(dets[3].item<float>()*100));
    send_list.push_back((char)(dets[4].item<float>()*100));
    send_list.push_back(126);
    uart.send(send_list);
//    std::cout << "send flag: " << send_flag << ", send list: " << send_list << std::endl;
}