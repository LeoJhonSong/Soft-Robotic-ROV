#ifndef MARKER_DETECTOR_H
#define MARKER_DETECTOR_H

#include <opencv2/highgui.hpp>
#include "aruco.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

namespace marker
{
    const char VER_OPENCV = 0;
    const char VER_ARUCO = 1;

    const char MODE_DETECT = 2;
    const char MODE_TRACK = 3;
    
    const std::map<int, cv::Point2f> MARKER_OFFSETS = {
        {39, cv::Point2f(120, 110)},
        {35, cv::Point2f(-120, 160)}
    };

    struct MarkerInfo
    {
        int id;
        cv::Point2f center;                     //中心点的像素坐标
        std::vector<cv::Point2f> corners;       //Marker的四个corner的像素坐标
        cv::Mat Rvec;                           //三轴旋转(弧度值)
        cv::Mat Tvec;                           //三轴位移
        MarkerInfo()
        {
            id = -1;
            center = cv::Point2f(0.0, 0.0);
            Rvec = (cv::Mat_<float>(3, 1) << 0, 0, 0);
            Tvec = (cv::Mat_<float>(3, 1) << 0, 0, 0);
        }
        MarkerInfo(int _id, cv::Point2f _center)
        {
            id = _id;
            center = _center;
            Rvec = (cv::Mat_<float>(3, 1) << 0, 0, 0);
            Tvec = (cv::Mat_<float>(3, 1) << 0, 0, 0);
        }
    };

    class MarkerDetector
    {
    private:
        // 相机参数
        const cv::Mat INTRINSICS_DEFAULT = (cv::Mat_<double>(3, 3) <<
                                            420.019, 0.0, 330.8676,
                                            0.0,419.6044, 217.8731,
                                            0.0, 0.0, 1.0);
        const cv::Mat DISTCOEFFS_DEFAULT = (cv::Mat_<float>(5, 1) << 0.1431, -0.4943, 0, 0, 0);
        const cv::Size IMG_SIZE_DEFAULT = cv::Size(640, 360);
        // marker每个格子的大小 单位: 米
        const double MARK_SIZE_DEFAULT = 0.05;
        // marker dictionary
        const char* DICT_DEFAULT = "ARUCO_MIP_36h12";
        const cv::aruco::PREDEFINED_DICTIONARY_NAME DICT_O_DEFAULT = cv::aruco::DICT_5X5_250;

        // marker每个格子的大小 单位: 米
        double marker_size;
        //相机参数
        aruco::CameraParameters camera;
        // 检测器 --aruco version
        aruco::MarkerDetector detector_a;
        // 跟踪器
        std::map<uint32_t, aruco::MarkerPoseTracker> MTracker;
        // 字典 opencv版用
        cv::Ptr<cv::aruco::Dictionary> dictionary_o;
    
        // 方法
        // 检测所有marker
        std::vector<marker::MarkerInfo> _detect_markers_aruco(cv::Mat &img, bool visible, bool track);
        std::vector<marker::MarkerInfo> _detect_markers_opencv(cv::Mat &img, bool visible);
        // 跟踪所有maker
        std::vector<marker::MarkerInfo> _track_markers_aruco(cv::Mat &img, bool visible);

    public:
        // MarkerDetector();MarkerDetector
        MarkerDetector(double marker_size=0.05, const char* dict="ARUCO_MIP_36h12")
        {
            // detect --aruco version 
            this->camera.CameraMatrix = this->INTRINSICS_DEFAULT;
            this->camera.Distorsion = this->DISTCOEFFS_DEFAULT;
            this->camera.CamSize = this->IMG_SIZE_DEFAULT;

            // 此语句集成到比赛工程中会在编译时报错,注释掉之后不影响O版代码的使用
            // this->detector_a.setDictionary(dict);
            
            this->marker_size = marker_size;
            this->dictionary_o = cv::aruco::getPredefinedDictionary(this->DICT_O_DEFAULT);
        }
        MarkerDetector(aruco::CameraParameters camera, double marker_size=0.05, const char* dict="ARUCO_MIP_36h12")
        {
            // detect --aruco version
            this->camera = camera;
            // this->detector_a.setDictionary(dict);
            this->marker_size = marker_size;
            this->dictionary_o = cv::aruco::getPredefinedDictionary(this->DICT_O_DEFAULT);
        }
        MarkerDetector(cv::Size camera_size, double marker_size=0.05, const char* dict="ARUCO_MIP_36h12")
        {
            // detect --aruco version
            this->camera.CameraMatrix = this->INTRINSICS_DEFAULT;
            this->camera.Distorsion = this->DISTCOEFFS_DEFAULT;
            this->camera.CamSize = camera_size;
            // this->detector_a.setDictionary(dict);
            this->marker_size = marker_size;
            this->dictionary_o = cv::aruco::getPredefinedDictionary(this->DICT_O_DEFAULT);
        }
        ~MarkerDetector(){}

        void camera_resize(cv::Size new_size);
        void set_dict_o(cv::aruco::PREDEFINED_DICTIONARY_NAME dict_o);
        marker::MarkerInfo detect_single_marker(cv::Mat &img, bool visible=true, char ver=marker::VER_ARUCO, char mode=marker::MODE_DETECT);
        marker::MarkerInfo detect_average_marker(cv::Mat &img, bool visible=true, char ver=marker::VER_ARUCO, char mode=marker::MODE_DETECT);
        std::vector<marker::MarkerInfo> detect_markers(cv::Mat &img, bool visible=true, char ver=marker::VER_ARUCO, char mode=marker::MODE_DETECT);
    };
};


#endif