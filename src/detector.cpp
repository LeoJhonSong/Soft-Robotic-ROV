//
// Created by sean on 7/17/19.
//

#include "detector.h"

Detector::Detector(unsigned int num_classes, int top_k,
        float nms_thresh, unsigned int tub, int ssd_dim){
    this->num_classes = num_classes;
    this->top_k = top_k;
    this->nms_thresh = nms_thresh;
    this->tub = tub;
    this->ssd_dim = ssd_dim;
    this->output = torch::zeros({1, this->num_classes, this->top_k, 5}, torch::kFloat).to(torch::kCUDA);
    this->log_params();
}

void Detector::log_params(){
    std::cout << "num_classes: " << num_classes << std::endl;
    std::cout << "top_k: " << top_k << std::endl;
    std::cout << "nms_thresh: " << nms_thresh << std::endl;
    std::cout << "tub: " << tub << std::endl;
    std::cout << "ssd_dim: " << ssd_dim << std::endl;
    std::cout << "out size: " << output.sizes() << output.dtype() << std::endl;
}

torch::Tensor Detector::detect(const torch::Tensor& loc, const torch::Tensor& conf, std::vector<float> conf_thresh){
    this->output.zero_();
    torch::Tensor c_mask, l_mask, boxes, keep, nms_score, nms_box;
    for(unsigned int cl=1; cl<this->num_classes; cl++){
        clock_t t = clock();
        c_mask = conf[cl].gt(conf_thresh.at(cl-1));
        clock_t t1 = clock();
        if(c_mask.sum().item<float_t >() == 0){
            continue;
        }
        clock_t t2 = clock();
        torch::Tensor scores = conf[cl].masked_select(c_mask);
        l_mask = c_mask.unsqueeze(1).expand_as(loc);
        boxes = loc.masked_select(l_mask).view({-1, 4});
        std::tuple<torch::Tensor, int> nms_result = nms(boxes.mul(this->ssd_dim), scores);
        keep = std::get<0>(nms_result);
        int count = std::get<1>(nms_result);

        nms_score = scores.index_select(0, keep);
        nms_box = boxes.index_select(0, keep);
//        torch::Tensor det_result = torch::cat({nms_score.unsqueeze(1), nms_box}, 1);
        this->output[0][cl].slice(0, 0, count) = torch::cat({nms_score.unsqueeze(1), nms_box}, 1);
        std::cout << "size: " << scores.sizes() << ", nms: " << (t2 - t1) * 1.0 / CLOCKS_PER_SEC * 1000
                  << ", time: " << (clock() - t) * 1.0 / CLOCKS_PER_SEC * 1000 << std::endl;
    }

    return this->output;

}

std::tuple<torch::Tensor, int> Detector::nms(const torch::Tensor& boxes, const torch::Tensor& scores){
    torch::Tensor keep = torch::zeros(scores.sizes()).to(torch::kLong).to(torch::kCUDA);
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
    torch::Tensor xx1, xx2, yy1, yy2, w, h, inters, rem_areas, unions, IoU;
    while(idx.numel() > 0){
        int i = idx[idx.size(0)-1].item<int32_t >();
        keep[count++] = i;
        if(idx.size(0) == 1) break;
        idx = idx.slice(0, 0, idx.size(0)-1);
        xx1 = x1.index_select(0, idx).clamp_min(x1[i].item<int32_t >());
        xx2 = x2.index_select(0, idx).clamp_max(x2[i].item<int32_t >());
        yy1 = y1.index_select(0, idx).clamp_min(y1[i].item<int32_t >());
        yy2 = y2.index_select(0, idx).clamp_max(y2[i].item<int32_t >());
        w = (xx2 - xx1).clamp_min(0);
        h = (yy2 - yy1).clamp_min(0);
        inters = w * h;
        rem_areas = area.index_select(0, idx);
        unions = (rem_areas - inters) + area[i];
        IoU = inters / unions;
        idx = idx.masked_select(IoU.le(this->nms_thresh));
    }
    std::get<0>(nms_result) = keep.slice(0, 0, count);
    std::get<1>(nms_result) = count;
//    std::cout << std::get<0>(nms_result) << ", " << std::get<1>(nms_result) <<std::endl;
    return nms_result;
//    std::cout << keep.sizes() << ", " << count.sizes() << ", " << x1.sizes() << area.sizes()<< std::endl;
}

void Detector::visualization(cv::Mat img, const torch::Tensor& detections){
    for(unsigned int j=1; j<detections.size(1); j++){
        torch::Tensor dets = detections[0][j];
        if(dets.sum().item<float_t>() == 0) continue;
        torch::Tensor mask = dets.slice(1, 0, 1).gt(0.0).expand_as(dets);
        dets = dets.masked_select(mask);
        dets = dets.view({dets.size(0)/mask.size(1), mask.size(1)});
        torch::Tensor boxes = dets.slice(1, 1, 5);
        boxes.slice(1, 0, 1) *= img.cols;
        boxes.slice(1, 2, 3) *= img.cols;
        boxes.slice(1, 1, 2) *= img.rows;
        boxes.slice(1, 3, 4) *= img.rows;
        torch::Tensor scores = dets.slice(1, 0, 1).squeeze(1);
        for(unsigned i=0; i<scores.size(0); i++){
            int x1 = boxes[i][0].item<int32_t>();
            int y1 = boxes[i][1].item<int32_t>();
            int x2 = boxes[i][2].item<int32_t>();
            int y2 = boxes[i][3].item<int32_t>();
            if(((x2-x1)*(y2-y1))/(img.cols*img.rows)>0.1) continue;
//            cv::Rect rect(x1, y1, x2, y2);
            cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), this->color.at(j-1), 1, 1, 0);
        }
    }
    cv::imshow("ResDet", img);
    cv::waitKey(1);
}

