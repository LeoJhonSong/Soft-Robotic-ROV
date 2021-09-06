#include "marker_detector.h"

void marker::MarkerDetector::camera_resize(cv::Size new_size)
{
    this->camera.CamSize = new_size;
}
void marker::MarkerDetector::set_dict_o(cv::aruco::PREDEFINED_DICTIONARY_NAME dict_o)
{
    this->dictionary_o = cv::aruco::getPredefinedDictionary(dict_o);
}

marker::MarkerInfo marker::MarkerDetector::detect_single_marker(cv::Mat &img, bool visible, char ver, char mode)
{
    if (ver == marker::VER_ARUCO)
    {
        if (mode == marker::MODE_DETECT)
        {
            std::vector<marker::MarkerInfo> marker_infos = this->_detect_markers_aruco(img, visible, false);
            if (marker_infos.size() > 0)
            {
                return marker_infos.at(0);
            }
            else
            {
                return marker::MarkerInfo();
            }
        }
        else if (mode == marker::MODE_TRACK)
        {
            std::vector<marker::MarkerInfo> marker_infos = this->_detect_markers_aruco(img, visible, true);
            if (marker_infos.size() > 0)
            {
                return marker_infos.at(0);
            }
            else
            {
                return marker::MarkerInfo();
            }
        }
        else
        {
            std::cout << "\"mode\" is wrong " << std::endl;
        }
    }
    else if (ver == marker::VER_OPENCV)
    {
        if (mode == marker::MODE_DETECT)
        {
            std::vector<marker::MarkerInfo> marker_infos = this->_detect_markers_opencv(img, visible);
            if (marker_infos.size() > 0)
            {
                return marker_infos.at(0);
            }
            else
            {
                return marker::MarkerInfo();
            }
        }
        else
        {
            std::cout << "VER_OPENCV only supports MODE_DETECT" << std::endl;
        }
    }
    else
    {
        std::cout << "\"ver\" is wrong " << std::endl;
    }
    return marker::MarkerInfo();
}

std::vector<marker::MarkerInfo> marker::MarkerDetector::detect_markers(cv::Mat &img, bool visible, char ver, char mode)
{
    if (ver == marker::VER_ARUCO)
    {
        if (mode == marker::MODE_DETECT)
        {
            return this->_detect_markers_aruco(img, visible, false);
        }
        else if (mode == marker::MODE_TRACK)
        {
            return this->_detect_markers_aruco(img, visible, true);
        }
        else
        {
            std::cout << "\"mode\" is wrong " << std::endl;
        }
    }
    else if (ver == marker::VER_OPENCV)
    {
        if (mode == marker::MODE_DETECT)
        {
            return this->_detect_markers_opencv(img, visible);
        }
        else
        {
            std::cout << "VER_OPENCV only supports MODE_DETECT" << std::endl;
        }
    }
    else
    {
        std::cout << "\"ver\" is wrong " << std::endl;
    }
    return std::vector<marker::MarkerInfo>();
}

// marker::MarkerInfo marker::MarkerDetector::detect_average_marker(cv::Mat &img, bool visible, char ver, char mode)
// {
//     std::vector<marker::MarkerInfo> resultes = detect_markers(img, visible, ver, mode);
//     if (resultes.empty())
//     {
//         return marker::MarkerInfo();
//     }
//     cv::Point2f marker_centers_average(0, 0);
//     int marker_num = 0;
//     for (int i = 0; i < resultes.size(); i++)
//     {
//         if (MARKER_OFFSETS.count(resultes.at(i).id) <= 0)
//             continue;
//         marker_num++;
//         cv::Point2f true_center = (resultes.at(i).center + marker::MARKER_OFFSETS.at(resultes.at(i).id));
//         marker_centers_average.x += true_center.x;
//         marker_centers_average.y += true_center.y;
//         cv::circle(img, true_center, 6, cv::Scalar(0, 0, 255), 1, 8, 0);
//     }
//     marker_centers_average.x /= marker_num;
//     marker_centers_average.y /= marker_num;
//     return marker::MarkerInfo(1, marker_centers_average);
// }
// 检测所有marker --aruco version
std::vector<marker::MarkerInfo> marker::MarkerDetector::_detect_markers_aruco(cv::Mat &img, bool visible, bool track)
{
    std::vector<marker::MarkerInfo> detected_markers;
    auto markers = this->detector_a.detect(img, this->camera, 0.1); // 0.05 is the marker size
    for (auto &marker : markers)
    {
        if (track)
        {
            this->MTracker[marker.id].estimatePose(marker, this->camera, this->marker_size); // call its tracker and estimate the pose
        }
        marker::MarkerInfo marker_info;
        marker_info.id = marker.id;
        marker_info.Rvec = marker.Rvec;
        marker_info.Tvec = marker.Tvec;
        // 中心点
        float c_x = 0;
        float c_y = 0;
        for (int i = 0; i < marker.size(); i++)
        {
            c_x += marker[i].x;
            c_y += marker[i].y;
            // 存储corners
            marker_info.corners.push_back(marker[i]);
        }
        c_x /= marker.size();
        c_y /= marker.size();
        marker_info.center = cv::Point2f(c_x, c_y);
        detected_markers.push_back(marker_info);

        if (visible)
        {
            // 可视化
            // 坐标轴
            aruco::CvDrawingUtils::draw3dAxis(img, marker, this->camera);
            cv::circle(img, cv::Point2f(c_x, c_y), 6, cv::Scalar(0, 0, 255), -1, 8, 0);
        }
        // std::cout << m[0] << std::endl;
        // std::cout << m.Rvec << " " << m.Tvec << std::endl;
    }
    return detected_markers;
}

// 检测所有marker --opencv version
std::vector<marker::MarkerInfo> marker::MarkerDetector::_detect_markers_opencv(cv::Mat &img, bool visible)
{
    std::vector<marker::MarkerInfo> detected_markers;

    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners;
    cv::aruco::detectMarkers(img, this->dictionary_o, corners, ids); //检测靶标
    if (ids.size() > 0)
    {
        std::vector<cv::Vec3d> rvecs, tvecs;
        cv::aruco::estimatePoseSingleMarkers(corners, this->marker_size, this->camera.CameraMatrix, this->camera.Distorsion, rvecs, tvecs); //求解旋转矩阵rvecs和平移矩阵tvecs
        for (int i = 0; i < ids.size(); i++)
        {
            marker::MarkerInfo marker_info;
            marker_info.id = ids.at(i);
            marker_info.corners = corners.at(i);
            marker_info.Rvec = cv::Mat(rvecs.at(i));
            marker_info.Tvec = cv::Mat(tvecs.at(i));
            float c_x = 0;
            float c_y = 0;
            for (auto p : corners[i])
            {
                c_x += p.x;
                c_y += p.y;
            }
            c_x /= corners[i].size();
            c_y /= corners[i].size();
            marker_info.center = cv::Point2f(c_x, c_y);
            detected_markers.push_back(marker_info);

            if (visible)
            {
                cv::circle(img, cv::Point2f(c_x, c_y), 6, cv::Scalar(0, 0, 255), -1, 8, 0);
                cv::aruco::drawDetectedMarkers(img, corners, ids); //绘制检测到的靶标的框
                cv::aruco::drawAxis(img, this->camera.CameraMatrix, this->camera.Distorsion, rvecs[i], tvecs[i], 0.1);
            }
        }
    }
    return detected_markers;
}

// 跟踪所有marker
std::vector<marker::MarkerInfo> marker::MarkerDetector::_track_markers_aruco(cv::Mat &img, bool visible)
{
    std::vector<marker::MarkerInfo> tracked_markers;
    std::vector<aruco::Marker> Markers = this->detector_a.detect(img);
    for (auto &marker : Markers) // for each marker
    {
        this->MTracker[marker.id].estimatePose(marker, this->camera, this->marker_size); // call its tracker and estimate the pose

        marker::MarkerInfo marker_info;
        marker_info.id = marker.id;
        marker_info.Rvec = marker.Rvec;
        marker_info.Tvec = marker.Tvec;
        // 中心点
        float c_x = 0;
        float c_y = 0;
        for (int i = 0; i < marker.size(); i++)
        {
            c_x += marker[i].x;
            c_y += marker[i].y;
            // 存储corners
            marker_info.corners.push_back(marker[i]);
        }
        c_x /= marker.size();
        c_y /= marker.size();
        marker_info.center = cv::Point2f(c_x, c_y);
        tracked_markers.push_back(marker_info);
    }

    if (visible)
    {
        // for each marker, draw info and its boundaries in the image
        for (unsigned int i = 0; i < Markers.size(); i++)
        {
            // std::cout << Markers[i] << std::endl;
            Markers[i].draw(img, cv::Scalar(0, 0, 255), 2);
        }
        // draw a 3d cube in each marker if there is 3d info
        if (this->camera.isValid() && this->marker_size != -1)
        {
            for (unsigned int i = 0; i < Markers.size(); i++)
            {
                if (Markers[i].isPoseValid())
                {
                    aruco::CvDrawingUtils::draw3dCube(img, Markers[i], this->camera);
                    aruco::CvDrawingUtils::draw3dAxis(img, Markers[i], this->camera);
                }
            }
        }
    }
    return tracked_markers;
}