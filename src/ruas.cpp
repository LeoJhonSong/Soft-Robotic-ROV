//
// Created by sean on 19-7-22.
//

#include "ruas.h"

CFilt::CFilt(int row, int col, int clahe_thresh)
{
    this->row = row;
    this->col = col;
    this->trans_func = cv::Mat::zeros(row, col, CV_32FC1);
    this->wiener_filter = cv::Mat::zeros(row, col, CV_32FC1);
    this->zeros_gpu.upload(cv::Mat::zeros(row, col, CV_32FC1));
    this->claher_gpu = cv::cuda::createCLAHE();
    this->claher_gpu->setClipLimit(clahe_thresh);
    this->dft_gpu = cv::cuda::createDFT(cv::Size(col, row), cv::DFT_COMPLEX_INPUT);
    this->dft_inv_gpu = cv::cuda::createDFT(cv::Size(col, row), cv::DFT_COMPLEX_INPUT | cv::DFT_INVERSE);
}

CFilt::~CFilt()
{
    this->trans_func.release();
    this->wiener_filter.release();
    this->zeros_gpu.release();
    this->claher_gpu.release();
    this->dft_gpu.release();
    this->dft_inv_gpu.release();
}

void CFilt::get_wf(float k, float r)
{

    int nRows = this->row;
    int nCols = this->col;
    int u = 0;
    int v = 0;

    if (this->trans_func.isContinuous())
    {
        nCols *= this->row;
        float *data = this->trans_func.ptr<float>(0);

        for (int p = 0; p < nCols; ++p)
        {
            v = p / (this->trans_func.step / 4);
            u = p % (this->trans_func.step / 4);
            data[p] = exp(-k * pow(10, -6) * pow(pow(u - this->col / 2.0, 2) + pow(v - this->row / 2.0, 2), 5.0 / 6));
        }
    }
    else
    {
        for (v = 0; v < nRows; ++v)
        {
            float *data = this->trans_func.ptr<float>(v);
            for (u = 0; u < nCols; ++u)
            {
                data[u] = exp(-(float)k * pow(10, -6) * pow(pow(u - this->col / 2.0, 2) + pow(v - this->row / 2.0, 2), 5.0 / 6));
            }
        }
    }
    this->wiener_filter = this->trans_func.mul(this->trans_func) / (this->trans_func.mul(this->trans_func) + r * 0.1) / this->trans_func;
    cv::Mat wiener_filter_ele[] = {this->wiener_filter, this->wiener_filter};
    cv::merge(wiener_filter_ele, 2, this->wiener_filter);
    this->wiener_filter_gpu.upload(this->wiener_filter);
}

void CFilt::wiener_gpu(cv::Mat &frame)
{

    std::vector<cv::cuda::GpuMat> BGR;
    std::vector<cv::cuda::GpuMat> BGR_ds_split;
    cv::cuda::split(frame, BGR);
    cv::cuda::GpuMat BGR_dft, BGR_comp, BGR_comp2, BGR_ds, BGR_fi;
    for (unsigned int i = 0; i < BGR.size(); i++)
    {
        cv::cuda::normalize(BGR[i], BGR[i], 0, 1, cv::NORM_MINMAX, CV_32F);
        cv::cuda::GpuMat BGR_ele[] = {BGR[i], zeros_gpu};
        cv::cuda::merge(BGR_ele, 2, BGR_comp);
        dft_gpu->compute(BGR_comp, BGR_dft);
        cv::cuda::multiply(BGR_dft, this->wiener_filter_gpu, BGR_dft, 1, -1);
        dft_inv_gpu->compute(BGR_dft, BGR_ds);
        cv::cuda::split(BGR_ds, BGR_ds_split);
        BGR[i] = BGR_ds_split.front();
        cv::cuda::normalize(BGR[i], BGR[i], 0, 255, cv::NORM_MINMAX, CV_8U);
        claher_gpu->apply(BGR[i], BGR[i]);
    }

    cv::cuda::merge(BGR, frame);
}

void CFilt::clahe_gpu(cv::Mat &frame)
{

    std::vector<cv::cuda::GpuMat> BGR;
    cv::cuda::split(frame, BGR);
    for (unsigned int i = 0; i < BGR.size(); i++)
    {
        claher_gpu->apply(BGR[i], BGR[i]);
    }

    cv::cuda::merge(BGR, frame);
}