//
// Created by sean on 7/17/19.
//

#include "detector.h"
#include "color.h"
#include<cmath>

extern bool save_a_frame;
extern std::queue<cv::Mat> det_frame_queue;
extern std::queue<std::pair<cv::Mat, int>> img_queue;
extern float max_depth, curr_depth;
extern int send_byte;
extern bool detect_scallop;
extern const float GRAP_THRESH_XC;
extern const float GRAP_THRESH_XW;
extern const float GRAP_THRESH_YC;
extern const float GRAP_THRESH_YW;
const int MATCHED_TIMES_THRESH = 15;

Detector::~Detector()=default;


Detector::Detector(unsigned int num_classes, int top_k, float nms_thresh, unsigned char tub, int ssd_dim, bool track){
    this->num_classes = num_classes;
    this->top_k = top_k;
    this->nms_thresh = nms_thresh;
    this->tub = tub;
    this->ssd_dim = ssd_dim;
    this->small_size_filter = 0.005;
    this->large_size_filter = 0.1;
    this->y_max_filter = 0.9;
    this->track_cl = 0;
    this->track_id = -1;
    this->frame_num = 0;
    this->track = false;
    if(this->tub > 0) {
        this->track = track;
        this->init_tubelets();
        this->hold_len = 5;
    }
    this->output = torch::zeros({1, this->num_classes, this->top_k, 7}, torch::kFloat);
    this->log_params();

}


void Detector::log_params(){
    print(WHITE, "num_classes: " << num_classes);
    print(WHITE, "top_k: " << top_k);
    print(WHITE, "nms_thresh: " << nms_thresh);
    print(WHITE, "tub: " << tub);
    print(WHITE, "ssd_dim: " << ssd_dim);
    print(WHITE, "out size: " << output.sizes());
    print(WHITE, "small_size_filter: " << small_size_filter);
    print(WHITE, "large_size_filter: " << large_size_filter);
    print(WHITE, "tubelets class size: " << this->tubelets.size());
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
        std::tuple<torch::Tensor, int> nms_result = nms(boxes, scores);
        torch::Tensor keep = std::get<0>(nms_result);
        int count = std::get<1>(nms_result);

        torch::Tensor nms_score = scores.index_select(0, keep);
        torch::Tensor nms_box = boxes.index_select(0, keep);
        this->output[0][cl].slice(0, 0, count) = torch::cat({nms_score.unsqueeze(1), nms_box, torch::zeros({count, 1}).fill_(-1), torch::zeros({count, 1}).fill_(100)}, 1);
    }
//    return this->output;
}


void Detector::detect(const torch::Tensor& loc, const torch::Tensor& conf, std::vector<float> conf_thresh, float tub_thresh){
    this->output.zero_();
    for(unsigned char cl=1; cl<this->num_classes; cl++){
        torch::Tensor c_mask = conf[cl].gt(conf_thresh.at(cl-1));
        if(c_mask.sum().item<int >() == 0){
            this->replenish_tubelets(cl, 0);
            this->delete_tubelets(cl);
            continue;
        }
        torch::Tensor scores = conf[cl].masked_select(c_mask);
        torch::Tensor l_mask = c_mask.unsqueeze(1).expand_as(loc);
        torch::Tensor boxes = loc.masked_select(l_mask).view({-1, 4});
        std::tuple<torch::Tensor, int> nms_result = nms(boxes, scores);
        torch::Tensor keep = std::get<0>(nms_result);
        int count = std::get<1>(nms_result);

        torch::Tensor nms_score = scores.index_select(0, keep);
        torch::Tensor nms_box = boxes.index_select(0, keep);
        torch::Tensor identity = torch::zeros({count}).fill_(-1);
        torch::Tensor matched_times = torch::zeros({count});
        if(count == 0){
            this->replenish_tubelets(cl, 0);
            this->delete_tubelets(cl);
            continue;
        }
        if(!this->tubelets.at(cl).empty()){
            torch::Tensor iou = this->iou(nms_box.mul(this->ssd_dim), cl);
            std::tuple<torch::Tensor,torch::Tensor> max_info = torch::max(iou, 1);
            torch::Tensor max_simi = std::get<0>(max_info);
            torch::Tensor max_idx = std::get<1>(max_info);
            torch::Tensor matched_mask = max_simi.gt(tub_thresh);
            for(unsigned char mt=0; mt<count; mt++) {
                if (matched_mask[mt].item<int>() > 0) {
                    identity[mt] = this->ides.at(cl).at(max_idx[mt].item<int>()).first;
                    matched_times[mt] = this->ides.at(cl).at(max_idx[mt].item<int>()).second + 1;
                }
            }
        }
        torch::Tensor new_id_mask = identity.eq(-1);
        if(new_id_mask.sum().item<int32_t >()>0){
            int current = this->history_max_ides[cl].item<int32_t >() + 1;
            torch::Tensor new_id = torch::arange(current, current + new_id_mask.sum().item<int >());
            this->history_max_ides[cl] = new_id[-1];
            int nid = 0;
            for(unsigned char m=0; m<identity.size(0); m++)
                if(new_id_mask[m].item<int >() > 0) identity[m] = new_id[nid++];
        }
        for(unsigned int tc=0; tc<count; tc++) {
            int curr_id = identity[tc].item<int>();
            if (this->tubelets.at(cl).find(curr_id) == this->tubelets.at(cl).end()) {
                this->tubelets.at(cl)[curr_id] = std::tuple<torch::Tensor, int, int>{nms_box[tc].unsqueeze(0).mul(this->ssd_dim), this->hold_len + 1, 0};
            } else {
                this->ides_set.at(cl).erase(curr_id);
                int id_matched_times = std::min(std::get<2>(this->tubelets.at(cl)[curr_id]) + 1, 100);
                this->tubelets.at(cl)[curr_id] = std::tuple<torch::Tensor, int, int>{nms_box[tc].unsqueeze(0).mul(this->ssd_dim), this->hold_len + 1, id_matched_times};
            }
        }
        this->output[0][cl].slice(0, 0, count) = torch::cat({nms_score.unsqueeze(1), nms_box, identity.unsqueeze(1), matched_times.unsqueeze(1)}, 1);
        this->replenish_tubelets(cl, count);
        int non_matched_size = 0;
//        for(auto s:this->ides_set.at(cl) ) {
//            torch::Tensor no_matched_box = std::get<0>(this->tubelets.at(cl)[s])[0];
//            if (no_matched_box.lt(0.1*this->ssd_dim).sum().item<uint8_t >()>0 || no_matched_box.gt(0.9*this->ssd_dim).sum().item<uint8_t >()>0 || std::get<2>(this->tubelets.at(cl)[s])<5 ) continue;
//            this->output[0][cl].slice(0, count + non_matched_size, count + non_matched_size+1) = torch::cat({torch::zeros({1}).fill_(0.01), std::get<0>(this->tubelets.at(cl)[s])[0].div(this->ssd_dim), torch::zeros({1}).fill_(s), torch::zeros({1}).fill_(std::get<2>(this->tubelets.at(cl)[s]))}, 0);
//            non_matched_size++;
//        }
        this->delete_tubelets(cl);
    }
//    return this->output;
}

void Detector::replenish_tubelets(unsigned char cl, int count){
    int non_matched_size = 0;
    for(auto s:this->ides_set.at(cl) ) {
        torch::Tensor no_matched_box = std::get<0>(this->tubelets.at(cl)[s])[0];
        if (no_matched_box.lt(0.1*this->ssd_dim).sum().item<uint8_t >()>0 || no_matched_box.gt(0.9*this->ssd_dim).sum().item<uint8_t >()>0 || std::get<2>(this->tubelets.at(cl)[s])<5 ) continue;
        this->output[0][cl].slice(0, count + non_matched_size, count + non_matched_size+1) = torch::cat({torch::zeros({1}).fill_(0.01), std::get<0>(this->tubelets.at(cl)[s])[0].div(this->ssd_dim), torch::zeros({1}).fill_(s), torch::zeros({1}).fill_(std::get<2>(this->tubelets.at(cl)[s]))}, 0);
        non_matched_size++;
    }
}


void Detector::detect_track(const torch::Tensor& loc, const torch::Tensor& conf, std::vector<float> conf_thresh){
    this->output.zero_();
    torch::Tensor prev_box = std::get<0>(this->tubelets.at(this->track_cl)[this->track_id]);
    torch::Tensor c_mask = conf[this->track_cl].gt(conf_thresh.at(this->track_cl-1));
    if(c_mask.sum().item<int >() == 0){
        this->output[0][this->track_cl].slice(0, 0, 1) = torch::cat({torch::zeros({1, 1}).fill_(0.01), std::get<0>(this->tubelets.at(this->track_cl)[this->track_id]).div(this->ssd_dim), torch::zeros({1, 1}).fill_(this->track_id), torch::zeros({1, 1}).fill_(std::get<2>(this->tubelets.at(this->track_cl)[this->track_id]))}, 1);
        this->delete_tubelets();
        return;
    }
    torch::Tensor scores = conf[this->track_cl].masked_select(c_mask);
    torch::Tensor l_mask = c_mask.unsqueeze(1).expand_as(loc);
    torch::Tensor boxes = loc.masked_select(l_mask).view({-1, 4});
    std::tuple<torch::Tensor, int> nms_result = prev_nms(boxes, scores, prev_box[0]);
    torch::Tensor keep = std::get<0>(nms_result);
    int count = std::get<1>(nms_result);

    if (count == 0){
        this->output[0][this->track_cl].slice(0, 0, 1) = torch::cat({torch::zeros({1, 1}).fill_(0.01), std::get<0>(this->tubelets.at(this->track_cl)[this->track_id]).div(this->ssd_dim), torch::zeros({1, 1}).fill_(this->track_id), torch::zeros({1, 1}).fill_(std::get<2>(this->tubelets.at(this->track_cl)[this->track_id]))}, 1);
        this->delete_tubelets();
        return;
    }
    torch::Tensor nms_score = scores.index_select(0, keep);
    torch::Tensor nms_box = boxes.index_select(0, keep);
    int id_matched_times = std::min(std::get<2>(this->tubelets.at(this->track_cl)[this->track_id]) + 1, 100);
    this->tubelets.at(this->track_cl)[this->track_id] = std::tuple<torch::Tensor, int, int>{nms_box[0].unsqueeze(0).mul(this->ssd_dim), this->hold_len + 1, id_matched_times};
    this->output[0][this->track_cl].slice(0, 0, count) = torch::cat({nms_score.unsqueeze(1), nms_box, torch::zeros({count, 1}).fill_(this->track_id), torch::zeros({count, 1}).fill_(id_matched_times)}, 1);
    this->delete_tubelets();

}


std::tuple<torch::Tensor, int> Detector::nms(torch::Tensor& boxes, torch::Tensor& scores){
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
//    torch::Tensor y;
    torch::Tensor area_mask = area.gt(this->small_size_filter) * area.lt(this->large_size_filter) * y2.lt(this->y_max_filter);
    x1 = x1.masked_select(area_mask);
    y1 = y1.masked_select(area_mask);
    x2 = x2.masked_select(area_mask);
    y2 = y2.masked_select(area_mask);
    area = area.masked_select(area_mask);
    boxes = torch::cat({x1.unsqueeze(-1), y1.unsqueeze(-1), x2.unsqueeze(-1), y2.unsqueeze(-1)}, 1);
    scores = scores.masked_select(area_mask);
    x1 *= this->ssd_dim;
    y1 *= this->ssd_dim;
    x2 *= this->ssd_dim;
    y2 *= this->ssd_dim;
    area *= this->ssd_dim * this->ssd_dim;

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


std::tuple<torch::Tensor, int> Detector::prev_nms(torch::Tensor& boxes, torch::Tensor& scores, const torch::Tensor& prev_box){
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
    torch::Tensor area_mask = area.gt(this->small_size_filter) * area.lt(this->large_size_filter);
    x1 = x1.masked_select(area_mask);
    y1 = y1.masked_select(area_mask);
    x2 = x2.masked_select(area_mask);
    y2 = y2.masked_select(area_mask);
    area = area.masked_select(area_mask);
    boxes = torch::cat({x1.unsqueeze(-1), y1.unsqueeze(-1), x2.unsqueeze(-1), y2.unsqueeze(-1)}, 1);
//    boxes = boxes.masked_select(area_mask.unsqueeze(-1).expand_as(boxes));
    scores = scores.masked_select(area_mask);
    x1 *= this->ssd_dim;
    y1 *= this->ssd_dim;
    x2 *= this->ssd_dim;
    y2 *= this->ssd_dim;
    area *= this->ssd_dim * this->ssd_dim;

    std::tuple<torch::Tensor, torch::Tensor> sorted_scores = scores.sort(0, false);
    torch::Tensor idx = std::get<1>(sorted_scores);
    // pre-box
    torch::Tensor area_tube = torch::mul(prev_box[2] - prev_box[0], prev_box[3] - prev_box[1]);
    torch::Tensor inners_tube = (x2.index_select(0, idx).clamp_max(prev_box[2].item<float>())-x1.index_select(0, idx).clamp_min(prev_box[0].item<float>())).clamp_min(0.0)
                                * (y2.index_select(0, idx).clamp_max(prev_box[3].item<float>())-y1.index_select(0, idx).clamp_min(prev_box[1].item<float>())).clamp_min(0.0);
    torch::Tensor unions_tube = area.index_select(0, idx) - inners_tube + area_tube;
    torch::Tensor IoU_tube = inners_tube.div(unions_tube);
    idx = idx.masked_select(IoU_tube.gt(0.3));
    if(idx.size(0) > this->top_k) idx = idx.slice(0, idx.size(0)-this->top_k, idx.size(0));
    while(idx.numel() > 0){
        int i = idx[-1].item<int >();
        keep[count++] = i;
        if(idx.size(0) == 1) break;
        idx = idx.slice(0, 0, idx.size(0)-1);
        torch::Tensor inners = (x2.index_select(0, idx).clamp_max(x2[i].item<float>()) - x1.index_select(0, idx).clamp_min(x1[i].item<float>())).clamp_min(0.0)
                               * (y2.index_select(0, idx).clamp_max(y2[i].item<float>()) - y1.index_select(0, idx).clamp_min(y1[i].item<float>())).clamp_min(0.0);
        torch::Tensor rem_areas = area.index_select(0, idx);
        torch::Tensor unions = (rem_areas - inners) + area[i];
        torch::Tensor IoU = inners / unions;
        idx = idx.masked_select(IoU.le(this->nms_thresh));
    }
    if (count == 0){
        std::get<0>(nms_result) = keep.slice(0, 0, count);
        std::get<1>(nms_result) = 0;
    }
    else {
        std::get<0>(nms_result) = keep[0];
        std::get<1>(nms_result) = 1;
    }
    return nms_result;
}


torch::Tensor Detector::iou(const torch::Tensor& boxes, unsigned char cl){
    std::map<int, std::tuple<torch::Tensor, int, int>> tubs = this->tubelets.at(cl);
    int tubs_size = tubs.size();
    torch::Tensor iou = torch::zeros({boxes.size(0), tubs_size});
    torch::Tensor x1 = boxes.slice(1, 0, 1).squeeze(-1);
    torch::Tensor y1 = boxes.slice(1, 1, 2).squeeze(-1);
    torch::Tensor x2 = boxes.slice(1, 2, 3).squeeze(-1);
    torch::Tensor y2 = boxes.slice(1, 3, 4).squeeze(-1);
    torch::Tensor area = torch::mul(x2-x1, y2-y1);
    unsigned char i = 0;
    for(auto& tube:tubs){
//        this->ides.at(cl).emplace_back(std::pair<int, int>{tube.first, std::get<2>(tube.second)});
//        this->ides_set.at(cl).insert(tube.first);
        torch::Tensor last_tube = std::get<0>(tube.second)[0];
        torch::Tensor area_tube = torch::mul(last_tube[2] - last_tube[0], last_tube[3] - last_tube[1]);
        torch::Tensor inner = (x2.clamp_max(last_tube[2].item<float>())-x1.clamp_min(last_tube[0].item<float>())).clamp_min(0.0)
                              * (y2.clamp_max(last_tube[3].item<float>())-y1.clamp_min(last_tube[1].item<float>())).clamp_min(0.0);
        torch::Tensor unions = area - inner + area_tube;
        iou.slice(1, i, i+1) = inner.div(unions).unsqueeze(-1);
        i++;
    }
    return iou;
}


std::vector<int> Detector::visualization(cv::Mat& img, std::ofstream& log_file){
    ++this->frame_num;
    std::stringstream stream;
    std::vector<int> loc;
    if (send_byte == 6) {
        int x1 = (int)(((float)this->send_list.at(1) / 100 - (float) this->send_list.at(3) / 100 / 2) * img.cols);
        int y1 = (int)(((float)this->send_list.at(2) / 100 - (float) this->send_list.at(4) / 100 / 2) * img.rows);
        int x2 = (int)(((float)this->send_list.at(1) / 100 + (float) this->send_list.at(3) / 100 / 2) * img.cols);
        int y2 = (int)(((float)this->send_list.at(2) / 100 + (float) this->send_list.at(4) / 100 / 2) * img.rows);
        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), this->color.at(0), 2, 1, 0);
    }
    else {
        if (this->track && this->track_cl > 0) {
            torch::Tensor dets = this->output[0][this->track_cl][0];
            torch::Tensor scores = dets[0];
            torch::Tensor ids = dets[5];
            int x1 = (dets[1] * img.cols).item<int>();
            int y1 = (dets[2] * img.rows).item<int>();
            int x2 = (dets[3] * img.cols).item<int>();
            int y2 = (dets[4] * img.rows).item<int>();
            loc = {(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1};
            cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), this->color.at(this->track_cl), 2, 1, 0);
            stream.str("");
            stream << std::fixed << std::setprecision(2) << scores.item<float>();
            cv::putText(img, std::to_string(ids.item<int>()) + ", " + stream.str(), cv::Point(x1, y1 - 5), 1, 1,
                        this->color.at(this->track_cl), 2);
            cv::putText(img, "track_id: " + std::to_string(this->track_id), cv::Point(img.cols - 120, 30), 1,
                        1, this->color.at(this->track_cl), 2);
            if (save_a_frame) {
                log_file << this->frame_num << ", " << (int) this->track_cl << ", " << std::setprecision(2)
                         << scores.item<float>() << ", " << dets[1].item<float>() << ", " << dets[2].item<float>()
                         << ", " << dets[3].item<float>() << ", " << dets[4].item<float>() << ", "
                         << ids.item<int>() << ", " << max_depth - curr_depth << std::endl;
            }
        } else {
            loc = {0, 0, 0, 0};
            for (unsigned char j = 1; j < this->num_classes; j++) {
                torch::Tensor dets = this->output[0][j];
                if (dets.sum().item<float>() == 0) continue;
                torch::Tensor score_mask = dets.slice(1, 0, 1).gt(0.0).expand_as(dets);
                torch::Tensor stable_mask = dets.slice(1, 6, 7).gt(5.0).expand_as(dets);
                torch::Tensor mask = score_mask * stable_mask;
                dets = dets.masked_select(mask);
                dets = dets.view({dets.size(0) / mask.size(1), mask.size(1)});
                torch::Tensor boxes = dets.slice(1, 1, 5);
                boxes.slice(1, 0, 1) *= img.cols;
                boxes.slice(1, 2, 3) *= img.cols;
                boxes.slice(1, 1, 2) *= img.rows;
                boxes.slice(1, 3, 4) *= img.rows;
                torch::Tensor scores = dets.slice(1, 0, 1).squeeze(1);
                torch::Tensor ids = dets.slice(1, 5, 6).squeeze(1);
                torch::Tensor matched_times = dets.slice(1, 6, 7).squeeze(1);
                for (unsigned char i = 0; i < boxes.size(0); i++) {
                    int id = ids[i].item<int>();
                    float score = scores[i].item<float>();
                    int x1 = boxes[i][0].item<int>();
                    int y1 = boxes[i][1].item<int>();
                    int x2 = boxes[i][2].item<int>();
                    int y2 = boxes[i][3].item<int>();
                    if ((y1 + y2)/2.0 > img.rows * 0.9) continue;
                    if (this->track && this->track_cl == 0 && score > 0.7 && matched_times[i].item<int>() > MATCHED_TIMES_THRESH) {
                        this->track_cl = j;
                        this->track_id = id;
                    }
                    cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), this->color.at(j), 2, 1, 0);
                    stream.str("");
                    stream << std::fixed << std::setprecision(2) << score;
                    cv::putText(img, std::to_string(id) + ", " + stream.str(), cv::Point(x1, y1 - 5), 1, 1,
                                this->color.at(j), 2);
                    this->stable_ides_set.at(j).insert(id);
                    if (save_a_frame) {
                        log_file << this->frame_num << ", " << (int) j << ", " << std::setprecision(2)
                                 << score << ", " << boxes[i][0].item<float>() / img.cols << ", "
                                 << boxes[i][1].item<float>() / img.rows << ", " << boxes[i][2].item<float>() / img.cols
                                 << ", " << boxes[i][3].item<float>() / img.rows << ", "
                                 << id << ", " << max_depth - curr_depth << std::endl;
                        print(BOLDWHITE, (boxes[i][0]+boxes[i][2]).item<float>() / 2 / img.cols << ", " << (boxes[i][1]+boxes[i][3]).item<float>() / 2 / img.rows );
                    }
                }
            }
        }
    }
    if (save_a_frame) {
        print(BOLDWHITE, "save detections for frame " << this->frame_num);
        img_queue.push(std::pair<cv::Mat, unsigned int>{img, this->frame_num});
    }
    if (this->tub > 0) {
        cv::putText(img, "trepang: " + std::to_string(this->stable_ides_set.at(1).size()), cv::Point(10, 30), 1,
                    1, this->color.at(1), 2);
        cv::putText(img, "urchin: " + std::to_string(this->stable_ides_set.at(2).size()), cv::Point(10, 45), 1,
                    1, this->color.at(2), 2);
        cv::putText(img, "shell: " + std::to_string(this->stable_ides_set.at(3).size()), cv::Point(10, 60), 1, 1,
                    this->color.at(3), 2);
        cv::putText(img, "starfish: " + std::to_string(this->stable_ides_set.at(4).size()), cv::Point(10, 75), 1,
                    1, this->color.at(4), 2);
    }
    cv::imshow("ResDet", img);
    det_frame_queue.push(img);
    save_a_frame = false;
    return loc;
}

int Detector::get_class_num(unsigned char cls){
    return this->stable_ides_set.at(cls).size();
}


void Detector::init_tubelets(){
    for(unsigned char i=0; i<num_classes; i++) {
        this->tubelets.emplace_back(std::map<int, std::tuple<torch::Tensor, int, int>>{});
        this->ides.emplace_back(std::vector<std::pair<int, int>>{});
        this->tubelets.at(i).clear();
        this->ides.at(i).clear();
        this->ides_set.emplace_back(std::set<int>{});
        this->ides_set.at(i).clear();
        this->stable_ides_set.emplace_back(std::set<int>{});
        this->stable_ides_set.at(i).clear();
    }
    this->history_max_ides = torch::zeros({num_classes}).fill_(-1);
}


void Detector::delete_tubelets(unsigned char cl){
    std::vector<int> delet_list;
//    std::map<int, std::tuple<torch::Tensor, int, int>> tubs = this->tubelets.at(cl);
    for(auto& tube:this->tubelets.at(cl)){
        if(--std::get<1>(tube.second) <= 0)
            delet_list.push_back(tube.first);
        std::get<1>(this->tubelets.at(cl)[tube.first]) = std::get<1>(tube.second);
    }
    for(auto id:delet_list)
        this->tubelets.at(cl).erase(id);
    this->ides.at(cl).clear();
    this->ides_set.at(cl).clear();
    for(auto& tube:this->tubelets.at(cl)) {
        this->ides.at(cl).emplace_back(std::pair<int, int>{tube.first, std::get<2>(tube.second)});
        this->ides_set.at(cl).insert(tube.first);
    }
}


void Detector::delete_tubelets(){
    std::vector<int> delet_list;
    for (unsigned char cl=1; cl<this->num_classes; ++cl) {
        delet_list.clear();
        std::map<int, std::tuple<torch::Tensor, int, int>> tubs = this->tubelets.at(cl);
        for (auto &tube:tubs) {
            if (--std::get<1>(tube.second) <= 0)
                delet_list.push_back(tube.first);
            std::get<1>(this->tubelets.at(cl)[tube.first]) = std::get<1>(tube.second);
        }
        for (auto id:delet_list)
            this->tubelets.at(cl).erase(id);
        this->ides.at(cl).clear();
        this->ides_set.at(cl).clear();
    }
}

std::vector<int> Detector::get_relative_position(Uart& uart){
    int selected_cls;
    std::vector<int> target_info;
    int detect_cls_num = detect_scallop ? 4 : 3;
    if (this->track_cl > 0 && this->track_cl < detect_cls_num)
        selected_cls = this->track_cl;
    if (selected_cls > this->num_classes-1){
        torch::Tensor scores = this->output[0].slice(1, 0, 1).slice(2, 0, 1);
        selected_cls = scores.argmax().item<unsigned char>();
    }
    torch::Tensor dets = this->output[0][selected_cls][0];
    target_info.clear();
    if (dets[0].item<float>()<0.3 || dets[6].item<int>()<5 || (dets[2] + dets[4]).item<float>()/2.0 > 0.9)
    {
        // 对应send_byte == 0
        target_info.push_back(1000);
        target_info.push_back(1000);
        return target_info;
    }
    // float dist = std::sqrt(std::pow(((dets[1].item<float>()+dets[3].item<float>())/2-0.5), 2) + std::pow(((dets[2].item<float>()+dets[4].item<float>())/4-1), 2));
    // xc, yc为0-1的数
    float xc = (dets[1].item<float>() + dets[3].item<float>()) / 2;
    float yc = (dets[2].item<float>() + dets[4].item<float>()) / 2;
    // 如果目标不在抓取阈值框内, 返回1
    if(std::abs(xc-GRAP_THRESH_XC) > GRAP_THRESH_XW || std::abs(yc-GRAP_THRESH_YC) > GRAP_THRESH_YW)
    {
        // 对应send_byte == 1
        target_info.push_back(2000);
        target_info.push_back(2000);
        return target_info;
    }
    target_info.push_back(selected_cls);  // 1为海参, 2为海胆, 3为扇贝
    target_info.push_back((xc *100));  // 为了让xc, yc为整数, 乘100
    target_info.push_back((yc *100));

    this->send_list.clear();
    this->send_list.push_back(selected_cls);  // 1为海参, 2为海胆, 3为扇贝
    this->send_list.push_back((char)(xc *100));  // 为了让xc, yc为整数, 乘100
    this->send_list.push_back((char)(yc *100));
    this->send_list.push_back((char)((dets[3].item<float>() - dets[1].item<float>()) *100));
    this->send_list.push_back((char)((dets[4].item<float>() - dets[2].item<float>()) *100));
    this->send_list.push_back((char)(round(max_depth/10.0)));

    return target_info;
}


std::vector<int> Detector::visual_detect(const torch::Tensor& loc, const torch::Tensor& conf, const std::vector<float>& conf_thresh, float tub_thresh, bool& reset, cv::Mat& img, std::ofstream& log_file){
    if(this->tub>0) {
        if(reset){
            this->reset_tracking_state();
            this->init_tubelets();
            reset = false;
        }
        if (this->track && this->track_cl > 0) {
            this->detect_track(loc, conf, conf_thresh);
            if (this->tubelets.at(this->track_cl).find(this->track_id) == this->tubelets.at(this->track_cl).end())
                this->reset_tracking_state();
        }
        else
            this->detect(loc, conf, conf_thresh, tub_thresh);
    }
    else this->detect(loc, conf, conf_thresh);
    // 置零所有框的扇贝的分数
    if (!detect_scallop) this->output[0][3] *= 0;

    return this->visualization(img, log_file);
}

void Detector::reset_tracking_state(){
    this->track_cl = 0;
    this->track_id = -1;
}