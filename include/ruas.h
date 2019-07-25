//
// Created by sean on 19-7-22.
//

#ifndef RESDET_RUAS_H
#define RESDET_RUAS_H
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>


class CFilt{

private:
    int row;
    int col;
    cv::Mat trans_func;
    cv::Mat wiener_filter;
    cv::cuda::GpuMat wiener_filter_gpu;
    cv::cuda::GpuMat zeros_gpu;
    cv::Ptr<cv::cuda::CLAHE> claher_gpu;
    cv::Ptr<cv::cuda::DFT> dft_gpu;
    cv::Ptr<cv::cuda::DFT> dft_inv_gpu;
    cv::cuda::Stream stream;
public:
    CFilt(int, int, int);
    void get_wf(float, float);
    void clahe_gpu(cv::Mat&);
    void wiener_gpu(cv::Mat&);
    ~CFilt();
};



#endif //RESDET_RUAS_H
