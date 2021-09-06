#include "detector.h"
#include "color.h"
#include <cmath>
#include <opencv2/highgui.hpp>

using namespace detector;

Detector::Detector(unsigned int num_classes, int top_k, float nms_thresh, bool tub, int ssd_dim, bool track)
{
    this->num_classes = num_classes;
    this->top_k = top_k;
    this->nms_thresh = nms_thresh;
    this->tub = tub;
    this->ssd_dim = ssd_dim;
    this->small_size_filter = 0.005;
    this->large_size_filter = 0.1;
    this->y_max_filter = 0.9;
    this->tracking_class = 0;
    this->track_id = -1;
    this->track = false;
    if (this->tub)
    {
        this->track = track;
        this->init_tubelets();
        this->hold_len = 5;
    }
    // the last dimension is: score, left edge, top edge, right edge, bottom edge, id, matched times
    // 最后一维度的7位分别是: score, 左边界, 右边界, 上边界, 下边界, id, 被识别到的次数
    this->candidates = torch::zeros({this->num_classes, this->top_k, 7}, torch::kFloat);
    this->log_params();
}

// print parameters of the detector
void Detector::log_params()
{
    // print(WHITE, "num_classes:          " << num_classes);
    // print(WHITE, "top_k:                " << top_k);
    // print(WHITE, "nms_thresh:           " << nms_thresh);
    // print(WHITE, "tub:                  " << tub);
    // print(WHITE, "ssd_dim:              " << ssd_dim);
    // print(WHITE, "out size:             " << candidates.sizes());
    // print(WHITE, "small_size_filter:    " << small_size_filter);
    // print(WHITE, "large_size_filter:    " << large_size_filter);
    // print(WHITE, "tubelets class size:  " << this->tubelets.size());
    print(BOLDMAGENTA, "[Detector] start");
}

// return number of candidates of the specific class
int Detector::get_class_num(unsigned char cls)
{
    return this->stable_id_set.at(cls).size();
}

// apply Non-Maximum Suppression
std::tuple<torch::Tensor, int> Detector::nms(torch::Tensor &boxes, torch::Tensor &scores)
{
    torch::Tensor keep = torch::zeros(scores.sizes()).to(torch::kLong);
    int count = 0;
    std::tuple<torch::Tensor, int> nms_result(keep, count);
    if (boxes.numel() == 0)
    {
        return nms_result;
    }
    torch::Tensor x1 = boxes.slice(1, 0, 1).squeeze(-1);
    torch::Tensor y1 = boxes.slice(1, 1, 2).squeeze(-1);
    torch::Tensor x2 = boxes.slice(1, 2, 3).squeeze(-1);
    torch::Tensor y2 = boxes.slice(1, 3, 4).squeeze(-1);
    torch::Tensor area = torch::mul(x2 - x1, y2 - y1);
    //    torch::Tensor y;
    torch::Tensor area_mask =
        area.gt(this->small_size_filter) * area.lt(this->large_size_filter) * y2.lt(this->y_max_filter);
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
    if (idx.size(0) > this->top_k)
        idx = idx.slice(0, idx.size(0) - this->top_k, idx.size(0));
    while (idx.numel() > 0)
    {
        int i = idx[idx.size(0) - 1].item<int>();
        keep[count++] = i;
        if (idx.size(0) == 1)
            break;
        idx = idx.slice(0, 0, idx.size(0) - 1);
        torch::Tensor inners = (x2.index_select(0, idx).clamp_max(x2[i].item<float>()) -
                                x1.index_select(0, idx).clamp_min(x1[i].item<float>()))
                                   .clamp_min(0.0) *
                               (y2.index_select(0, idx).clamp_max(y2[i].item<float>()) -
                                y1.index_select(0, idx).clamp_min(y1[i].item<float>()))
                                   .clamp_min(0.0);
        torch::Tensor rem_areas = area.index_select(0, idx);
        torch::Tensor unions = (rem_areas - inners) + area[i];
        torch::Tensor IoU = inners / unions;
        idx = idx.masked_select(IoU.le(this->nms_thresh));
    }
    std::get<0>(nms_result) = keep.slice(0, 0, count);
    std::get<1>(nms_result) = count;
    return nms_result;
}

std::tuple<torch::Tensor, int> Detector::prev_nms(torch::Tensor &boxes, torch::Tensor &scores,
                                                  const torch::Tensor &prev_box)
{
    torch::Tensor keep = torch::zeros(scores.sizes()).to(torch::kLong);
    int count = 0;
    std::tuple<torch::Tensor, int> nms_result(keep, count);
    if (boxes.numel() == 0)
    {
        return nms_result;
    }
    torch::Tensor x1 = boxes.slice(1, 0, 1).squeeze(-1);
    torch::Tensor y1 = boxes.slice(1, 1, 2).squeeze(-1);
    torch::Tensor x2 = boxes.slice(1, 2, 3).squeeze(-1);
    torch::Tensor y2 = boxes.slice(1, 3, 4).squeeze(-1);
    torch::Tensor area = torch::mul(x2 - x1, y2 - y1);
    torch::Tensor area_mask = area.gt(this->small_size_filter) * area.lt(this->large_size_filter);
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
    // pre-box
    torch::Tensor area_tube = torch::mul(prev_box[2] - prev_box[0], prev_box[3] - prev_box[1]);
    torch::Tensor inners_tube = (x2.index_select(0, idx).clamp_max(prev_box[2].item<float>()) -
                                 x1.index_select(0, idx).clamp_min(prev_box[0].item<float>()))
                                    .clamp_min(0.0) *
                                (y2.index_select(0, idx).clamp_max(prev_box[3].item<float>()) -
                                 y1.index_select(0, idx).clamp_min(prev_box[1].item<float>()))
                                    .clamp_min(0.0);
    torch::Tensor unions_tube = area.index_select(0, idx) - inners_tube + area_tube;
    torch::Tensor IoU_tube = inners_tube.div(unions_tube);
    idx = idx.masked_select(IoU_tube.gt(0.3));
    if (idx.size(0) > this->top_k)
        idx = idx.slice(0, idx.size(0) - this->top_k, idx.size(0));
    while (idx.numel() > 0)
    {
        int i = idx[-1].item<int>();
        keep[count++] = i;
        if (idx.size(0) == 1)
            break;
        idx = idx.slice(0, 0, idx.size(0) - 1);
        torch::Tensor inners = (x2.index_select(0, idx).clamp_max(x2[i].item<float>()) -
                                x1.index_select(0, idx).clamp_min(x1[i].item<float>()))
                                   .clamp_min(0.0) *
                               (y2.index_select(0, idx).clamp_max(y2[i].item<float>()) -
                                y1.index_select(0, idx).clamp_min(y1[i].item<float>()))
                                   .clamp_min(0.0);
        torch::Tensor rem_areas = area.index_select(0, idx);
        torch::Tensor unions = (rem_areas - inners) + area[i];
        torch::Tensor IoU = inners / unions;
        idx = idx.masked_select(IoU.le(this->nms_thresh));
    }
    if (count == 0)
    {
        std::get<0>(nms_result) = keep.slice(0, 0, count);
        std::get<1>(nms_result) = 0;
    }
    else
    {
        std::get<0>(nms_result) = keep[0];
        std::get<1>(nms_result) = 1;
    }
    return nms_result;
}

torch::Tensor Detector::iou(const torch::Tensor &boxes, unsigned char cl)
{
    std::map<int, std::tuple<torch::Tensor, int, int>> tubs = this->tubelets.at(cl);
    int tubs_size = tubs.size();
    torch::Tensor iou = torch::zeros({boxes.size(0), tubs_size});
    torch::Tensor x1 = boxes.slice(1, 0, 1).squeeze(-1);
    torch::Tensor y1 = boxes.slice(1, 1, 2).squeeze(-1);
    torch::Tensor x2 = boxes.slice(1, 2, 3).squeeze(-1);
    torch::Tensor y2 = boxes.slice(1, 3, 4).squeeze(-1);
    torch::Tensor area = torch::mul(x2 - x1, y2 - y1);
    unsigned char i = 0;
    for (auto &tube : tubs)
    {
        torch::Tensor last_tube = std::get<0>(tube.second)[0];
        torch::Tensor area_tube = torch::mul(last_tube[2] - last_tube[0], last_tube[3] - last_tube[1]);
        torch::Tensor inner =
            (x2.clamp_max(last_tube[2].item<float>()) - x1.clamp_min(last_tube[0].item<float>())).clamp_min(0.0) *
            (y2.clamp_max(last_tube[3].item<float>()) - y1.clamp_min(last_tube[1].item<float>())).clamp_min(0.0);
        torch::Tensor unions = area - inner + area_tube;
        iou.slice(1, i, i + 1) = inner.div(unions).unsqueeze(-1);
        i++;
    }
    return iou;
}

void Detector::init_tubelets()
{
    for (unsigned int i = 0; i < num_classes; i++)
    {
        this->tubelets.emplace_back(std::map<int, std::tuple<torch::Tensor, int, int>>{});
        this->ids.emplace_back(std::vector<std::pair<int, int>>{});
        this->tubelets.at(i).clear();
        this->ids.at(i).clear();
        this->id_set.emplace_back(std::set<int>{});
        this->id_set.at(i).clear();
        this->stable_id_set.emplace_back(std::set<int>{});
        this->stable_id_set.at(i).clear();
    }
    this->history_max_ides = torch::zeros({num_classes}).fill_(-1);
}

// 删除指定类型目标的轨迹
void Detector::delete_tubelet(unsigned int cl)
{
    std::vector<int> delete_list;
    //    std::map<int, std::tuple<torch::Tensor, int, int>> tubs = this->tubelets.at(cl);
    for (auto &tube : this->tubelets.at(cl))
    {
        if (--std::get<1>(tube.second) <= 0)
            delete_list.push_back(tube.first);
        std::get<1>(this->tubelets.at(cl)[tube.first]) = std::get<1>(tube.second);
    }
    for (auto id : delete_list)
        this->tubelets.at(cl).erase(id);
    this->ids.at(cl).clear();
    this->id_set.at(cl).clear();
    for (auto &tube : this->tubelets.at(cl))
    {
        this->ids.at(cl).emplace_back(std::pair<int, int>{tube.first, std::get<2>(tube.second)});
        this->id_set.at(cl).insert(tube.first);
    }
}

// 删除所有类型目标的轨迹
void Detector::delete_tubelets()
{
    std::vector<int> delete_list;
    for (unsigned int cl = 1; cl < this->num_classes; ++cl)
    {
        delete_list.clear();
        std::map<int, std::tuple<torch::Tensor, int, int>> tubs = this->tubelets.at(cl);
        for (auto &tube : tubs)
        {
            if (--std::get<1>(tube.second) <= 0)
                delete_list.push_back(tube.first);
            std::get<1>(this->tubelets.at(cl)[tube.first]) = std::get<1>(tube.second);
        }
        for (auto id : delete_list)
            this->tubelets.at(cl).erase(id);
        this->ids.at(cl).clear();
        this->id_set.at(cl).clear();
    }
}

void Detector::replenish_tubelets(unsigned char cl, int count)
{
    int non_matched_size = 0;
    for (auto s : this->id_set.at(cl))
    {
        torch::Tensor no_matched_box = std::get<0>(this->tubelets.at(cl)[s])[0];
        if (no_matched_box.lt(0.1 * this->ssd_dim).sum().item<uint8_t>() > 0 ||
            no_matched_box.gt(0.9 * this->ssd_dim).sum().item<uint8_t>() > 0 ||
            std::get<2>(this->tubelets.at(cl)[s]) < 5)
            continue;
        this->candidates[cl].slice(0, count + non_matched_size, count + non_matched_size + 1) =
            torch::cat({torch::zeros({1}).fill_(0.01), std::get<0>(this->tubelets.at(cl)[s])[0].div(this->ssd_dim),
                        torch::zeros({1}).fill_(s), torch::zeros({1}).fill_(std::get<2>(this->tubelets.at(cl)[s]))},
                       0);
        non_matched_size++;
    }
}

void Detector::reset_tracking_state()
{
    this->tracking_class = 0;
    this->track_id = -1;
}

// update this->candidates
void Detector::update(const torch::Tensor &loc, const torch::Tensor &conf, std::vector<float> conf_thresh)
{
    // clear candidates
    this->candidates.zero_();
    // traverse start from non-background class
    for (unsigned int current_class = 1; current_class < this->num_classes; current_class++)
    {
        torch::Tensor c_mask = conf[current_class].gt(conf_thresh.at(
            current_class)); // get a mask showing if each value in the tensor is greater than the threshold
        // if sum of the mask is zero, then no candidates with high confidence, skip current class
        if (c_mask.sum().item<int>() == 0)
            continue;
        torch::Tensor scores = conf[current_class].masked_select(c_mask);
        torch::Tensor l_mask = c_mask.unsqueeze(1).expand_as(loc);
        torch::Tensor boxes = loc.masked_select(l_mask).view({-1, 4});
        std::tuple<torch::Tensor, int> nms_result = nms(boxes, scores);
        torch::Tensor keep = std::get<0>(nms_result);
        int count = std::get<1>(nms_result);
        // do nms
        torch::Tensor nms_score = scores.index_select(0, keep);
        torch::Tensor nms_box = boxes.index_select(0, keep);
        this->candidates[current_class].slice(0, 0, count) = torch::cat(
            {nms_score.unsqueeze(1), nms_box, torch::zeros({count, 1}).fill_(-1), torch::zeros({count, 1}).fill_(100)},
            1);
    }
}

void Detector::update(const torch::Tensor &loc, const torch::Tensor &conf, std::vector<float> conf_thresh,
                      float tub_thresh)
{
    // clear candidates
    this->candidates.zero_();
    // traverse start from non-background class
    for (unsigned int current_class = 1; current_class < this->num_classes; current_class++)
    {
        torch::Tensor c_mask = conf[current_class].gt(
            conf_thresh.at(current_class)); // get a mask showing if each value in the tensor is greater than the
                                            // threshold if sum of the mask is zero, then no candidates with high
                                            // confidence, clear and skip current class
        if (c_mask.sum().item<int>() == 0)
        {
            this->replenish_tubelets(current_class, 0);
            this->delete_tubelet(current_class);
            continue;
        }
        torch::Tensor scores = conf[current_class].masked_select(c_mask);
        torch::Tensor l_mask = c_mask.unsqueeze(1).expand_as(loc);
        torch::Tensor boxes = loc.masked_select(l_mask).view({-1, 4});
        std::tuple<torch::Tensor, int> nms_result = nms(boxes, scores);
        torch::Tensor keep = std::get<0>(nms_result);
        int count = std::get<1>(nms_result);

        torch::Tensor nms_score = scores.index_select(0, keep);
        torch::Tensor nms_box = boxes.index_select(0, keep);
        torch::Tensor identity = torch::zeros({count}).fill_(-1);
        torch::Tensor matched_times = torch::zeros({count});
        if (count == 0)
        {
            this->replenish_tubelets(current_class, 0);
            this->delete_tubelet(current_class);
            continue;
        }
        if (!this->tubelets.at(current_class).empty())
        {
            torch::Tensor iou = this->iou(nms_box.mul(this->ssd_dim), current_class);
            std::tuple<torch::Tensor, torch::Tensor> max_info = torch::max(iou, 1);
            torch::Tensor max_simi = std::get<0>(max_info);
            torch::Tensor max_idx = std::get<1>(max_info);
            torch::Tensor matched_mask = max_simi.gt(tub_thresh);
            for (unsigned int mt = 0; mt < count; mt++)
            {
                if (matched_mask[mt].item<int>() > 0)
                {
                    identity[mt] = this->ids.at(current_class).at(max_idx[mt].item<int>()).first;
                    matched_times[mt] = this->ids.at(current_class).at(max_idx[mt].item<int>()).second + 1;
                }
            }
        }
        torch::Tensor new_id_mask = identity.eq(-1);
        if (new_id_mask.sum().item<int32_t>() > 0)
        {
            int current = this->history_max_ides[current_class].item<int32_t>() + 1;
            torch::Tensor new_id = torch::arange(current, current + new_id_mask.sum().item<int>());
            this->history_max_ides[current_class] = new_id[-1];
            int nid = 0;
            for (unsigned int m = 0; m < identity.size(0); m++)
                if (new_id_mask[m].item<int>() > 0)
                    identity[m] = new_id[nid++];
        }
        for (unsigned int tc = 0; tc < count; tc++)
        {
            int curr_id = identity[tc].item<int>();
            if (this->tubelets.at(current_class).find(curr_id) == this->tubelets.at(current_class).end())
            {
                this->tubelets.at(current_class)[curr_id] = std::tuple<torch::Tensor, int, int>{
                    nms_box[tc].unsqueeze(0).mul(this->ssd_dim), this->hold_len + 1, 0};
            }
            else
            {
                this->id_set.at(current_class).erase(curr_id);
                int id_matched_times = std::min(std::get<2>(this->tubelets.at(current_class)[curr_id]) + 1, 100);
                this->tubelets.at(current_class)[curr_id] = std::tuple<torch::Tensor, int, int>{
                    nms_box[tc].unsqueeze(0).mul(this->ssd_dim), this->hold_len + 1, id_matched_times};
            }
        }
        this->candidates[current_class].slice(0, 0, count) =
            torch::cat({nms_score.unsqueeze(1), nms_box, identity.unsqueeze(1), matched_times.unsqueeze(1)}, 1);
        this->replenish_tubelets(current_class, count);
        this->delete_tubelet(current_class);
    }
}

// update candidates and track specific class
void Detector::tracking_update(const torch::Tensor &loc, const torch::Tensor &conf, std::vector<float> conf_thresh)
{
    this->candidates.zero_();
    torch::Tensor prev_box = std::get<0>(this->tubelets.at(this->tracking_class)[this->track_id]);
    torch::Tensor c_mask = conf[this->tracking_class].gt(conf_thresh.at(this->tracking_class));
    // if sum of the mask is zero, then no candidates with high confidence, clear and skip current class
    if (c_mask.sum().item<int>() == 0)
    {
        this->candidates[this->tracking_class].slice(0, 0, 1) = torch::cat(
            {torch::zeros({1, 1}).fill_(0.01),
             std::get<0>(this->tubelets.at(this->tracking_class)[this->track_id]).div(this->ssd_dim),
             torch::zeros({1, 1}).fill_(this->track_id),
             torch::zeros({1, 1}).fill_(std::get<2>(this->tubelets.at(this->tracking_class)[this->track_id]))},
            1);
        this->delete_tubelets();
        return;
    }
    torch::Tensor scores = conf[this->tracking_class].masked_select(c_mask);
    torch::Tensor l_mask = c_mask.unsqueeze(1).expand_as(loc);
    torch::Tensor boxes = loc.masked_select(l_mask).view({-1, 4});
    std::tuple<torch::Tensor, int> nms_result = prev_nms(boxes, scores, prev_box[0]);
    torch::Tensor keep = std::get<0>(nms_result);
    int count = std::get<1>(nms_result);

    if (count == 0)
    {
        this->candidates[this->tracking_class].slice(0, 0, 1) = torch::cat(
            {torch::zeros({1, 1}).fill_(0.01),
             std::get<0>(this->tubelets.at(this->tracking_class)[this->track_id]).div(this->ssd_dim),
             torch::zeros({1, 1}).fill_(this->track_id),
             torch::zeros({1, 1}).fill_(std::get<2>(this->tubelets.at(this->tracking_class)[this->track_id]))},
            1);
        this->delete_tubelets();
        return;
    }
    torch::Tensor nms_score = scores.index_select(0, keep);
    torch::Tensor nms_box = boxes.index_select(0, keep);
    int id_matched_times = std::min(std::get<2>(this->tubelets.at(this->tracking_class)[this->track_id]) + 1, 100);
    this->tubelets.at(this->tracking_class)[this->track_id] = std::tuple<torch::Tensor, int, int>{
        nms_box[0].unsqueeze(0).mul(this->ssd_dim), this->hold_len + 1, id_matched_times};
    this->candidates[this->tracking_class].slice(0, 0, count) =
        torch::cat({nms_score.unsqueeze(1), nms_box, torch::zeros({count, 1}).fill_(this->track_id),
                    torch::zeros({count, 1}).fill_(id_matched_times)},
                   1);
    this->delete_tubelets();
}

std::vector<float> Detector::visualization(cv::Mat &img)
{
    std::stringstream stream;
    std::vector<float> loc;
    // when a target selected to be grasped, rectangle it
    // if track is enabled and tracking class is specified, draw box and put text for target from tracking class
    if (this->track && this->tracking_class > 0)
    {
        torch::Tensor target_info =
            this->candidates[this->tracking_class][0]; // select first one of tracking class as target
        torch::Tensor score = target_info[0];
        torch::Tensor id = target_info[5];
        float left = (target_info[1]).item<float>();
        float top = (target_info[2]).item<float>();
        float right = (target_info[3]).item<float>();
        float bottom = (target_info[4]).item<float>();
        loc = {(left + right) / 2, (top + bottom) / 2, right - left, bottom - top}; // cx, cy, width, height
        // scale the shape to size of frame
        left = left * (float)img.cols;
        top = top * (float)img.rows;
        right = right * (float)img.cols;
        bottom = bottom * (float)img.rows;
        cv::rectangle(img, cv::Point(left, top), cv::Point(right, bottom), this->color.at(this->tracking_class), 2, 1,
                      0);
        stream.str("");
        stream << std::fixed << std::setprecision(2) << score.item<float>();
        cv::putText(img, std::to_string(id.item<int>()) + ", " + stream.str(), cv::Point(left, top - 5), 1, 1,
                    this->color.at(this->tracking_class), 2);
        cv::putText(img, "track_id: " + std::to_string(this->track_id), cv::Point(img.cols - 120, 30), 1, 1,
                    this->color.at(this->tracking_class), 2);
    }
    // if track is not enabled, or no tracking target, draw and put text for filtered candidates and try to set
    // tracking_class, track_id
    else
    {
        loc = {0, 0, 0, 0};
        for (unsigned int i = 1; i < this->num_classes; i++)
        {
            torch::Tensor classed_candidates = this->candidates[i];
            // no candidates, skip
            if (classed_candidates.sum().item<float>() == 0)
                continue;
            // wired bug: scores become 0.01 without the following line. drop from 0.8/9 to 0.01.
            torch::Tensor score_mask = classed_candidates.slice(1, 0, 1).gt(0.01).expand_as(
                classed_candidates); // useless, candidates already filtered by confidence in update()
            torch::Tensor stable_mask = classed_candidates.slice(1, 6, 7).gt(5.0).expand_as(classed_candidates);
            torch::Tensor mask = score_mask * stable_mask;
            classed_candidates = classed_candidates.masked_select(mask);
            classed_candidates = classed_candidates.resize_({classed_candidates.size(0) / mask.size(1), mask.size(1)});
            torch::Tensor boxes = classed_candidates.slice(1, 1, 5);
            boxes.slice(1, 0, 1) *= img.cols; // left
            boxes.slice(1, 1, 2) *= img.rows; // top
            boxes.slice(1, 2, 3) *= img.cols; // right
            boxes.slice(1, 3, 4) *= img.rows; // bottom
            torch::Tensor scores = classed_candidates.slice(1, 0, 1).squeeze(1);
            torch::Tensor ids = classed_candidates.slice(1, 5, 6).squeeze(1);
            // 被识别到的次数
            torch::Tensor matched_times = classed_candidates.slice(1, 6, 7).squeeze(1);
            for (unsigned int j = 0; j < boxes.size(0); j++) // for every candidates in tracking class
            {
                int id = ids[j].item<int>();
                float score = scores[j].item<float>();
                int left = boxes[j][0].item<int>();
                int top = boxes[j][1].item<int>();
                int right = boxes[j][2].item<int>();
                int bottom = boxes[j][3].item<int>();
                if ((top + bottom) / 2.0 > img.rows * 0.9)
                    continue;
                if (this->track && this->tracking_class == 0 && matched_times[j].item<int>() > 30)
                {
                    this->tracking_class = i;
                    this->track_id = id;
                }
                // 绘制单个candidate的框和标签
                cv::rectangle(img, cv::Point(left, top), cv::Point(right, bottom), this->color.at(i), 2, 1, 0);
                stream.str("");
                stream << std::fixed << std::setprecision(2) << score;
                cv::putText(img, std::to_string(id) + ", " + stream.str(), cv::Point(left, top - 5), 1, 1,
                            this->color.at(i), 2);
                // 将当前candidate的id添加到stable_id_set
                this->stable_id_set.at(i).insert(id);
            }
        }
    }
    if (this->tub)
    {
        cv::putText(img, "trepang: " + std::to_string(this->stable_id_set.at(1).size()), cv::Point(10, 30), 1, 1,
                    this->color.at(1), 2);
        cv::putText(img, "urchin: " + std::to_string(this->stable_id_set.at(2).size()), cv::Point(10, 45), 1, 1,
                    this->color.at(2), 2);
        cv::putText(img, "scallop: " + std::to_string(this->stable_id_set.at(3).size()), cv::Point(10, 60), 1, 1,
                    this->color.at(3), 2);
        cv::putText(img, "starfish: " + std::to_string(this->stable_id_set.at(4).size()), cv::Point(10, 75), 1, 1,
                    this->color.at(4), 2);
    }
    return loc;
}

// for call from main
std::vector<float> Detector::detect_and_visualize(const torch::Tensor &loc, const torch::Tensor &conf,
                                                  const std::vector<float> &conf_thresh, float tub_thresh, bool &reset,
                                                  bool detect_scallop, cv::Mat &img)
{
    // update detection
    if (this->tub)
    {
        if (reset)
        {
            this->reset_tracking_state();
            this->init_tubelets();
            reset = false;
        }
        // if track is enabled and is tracking non-background target
        if (this->track && this->tracking_class > 0)
        {
            this->tracking_update(loc, conf, conf_thresh);
            if (this->tubelets.at(this->tracking_class).find(this->track_id) ==
                this->tubelets.at(this->tracking_class).end())
                this->reset_tracking_state();
        }
        else
            this->update(loc, conf, conf_thresh, tub_thresh);
    }
    else
        this->update(loc, conf, conf_thresh);
    // 若忽略扇贝, 置零所有框的扇贝的数据
    if (!detect_scallop)
        this->candidates[3] *= 0;
    // visualize
    return this->visualization(img);
}

Detector::~Detector() = default;