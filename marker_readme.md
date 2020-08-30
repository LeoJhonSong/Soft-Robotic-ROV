## aruco
### 安装
- 源码下载: https://sourceforge.net/projects/aruco/files/
- 安装:  
  ```
  解压
  mkdir build
  cd build
  cmake -D CMAKE_INSTALL_PREFIX=安装目录 ../
  make
  make install
  ```
### 手册地址
- https://docs.google.com/document/d/1QU9KoBtjSM2kF6ITOjQ76xqL7H0TEtXriJX5kwi9Kgc
### 注意
- 可以认为分两个版本:opencv集成版和aruco独立版本(下文简称A版和O版)
- CmakeLists中需要添加相应依赖
### marker生成
- 必须使用特定的marker才可以,A版和O版生成方式不同:
  1. O版: https://blog.csdn.net/zhou4411781/article/details/103262408, 目前代码中使用的是5x5大小的marker
  2. A版: 直接输出某个字典中的全部marker图片(一个字典记录了一定数量的marker,为检测器设置特定字典后,字典中有的marker才会被检测到)  
    (1) cd 编译后的build文件夹  
    (2) cd utils  
    (3) ./aruco_print_dictionary outdir dictionary [ -s bit_image_size: 75 default]  
      - outdir: 输出文件夹  
      - dictionary: 选择字典 可选值: ARUCO ARUCO_MIP_16h3 ARUCO_MIP_25h7 ARUCO_MIP_36h12 ARTOOLKITPLUS ARTOOLKITPLUSBCH TAG16h5 TAG25h7 TAG25h9 TAG36h11 TAG36h10 CHILITAGS ALL_DICTS 官方推荐: ARUCO_MIP_16h3  
      - bit_image_size: 图像大小 选填
### 代码说明
- 代码集成了三个版本:
  1. O版检测
  2. A版检测
  3. A版检测+跟踪: 多了一个跟踪器,手册上说可以增加鲁棒性
- 具体使用方法请看示范代码及注释
- 相机参数在marker_detector.h中设置
- 目前存在的问题: A版代码在单独的测试工程中可以使用,但是集成到比赛工程里后会出现问题,但是O版使用正常,可以先用着O版,感觉效果和A版差不多,都还不错
### 示范代码
```c++
#include "marker_detector.h"

void show_marker_info(marker::MarkerInfo &marker_info)
{
	// marker信息 含义详见marker_detector.h中的注释
	std::cout << "id: " << marker_info.id << std::endl;
	std::cout << "center: " << marker_info.center << std::endl;
	std::cout << "corners: " << marker_info.corners << std::endl;
	std::cout << "Rvec: " << marker_info.Rvec << std::endl;
	std::cout << "Tvec: " << marker_info.Tvec << std::endl;
}

const char SINGLE = 0;
const char ALL = 1;
const char MODE = SINGLE;

int main(int argc, char* argv[])
{
	cv::VideoCapture inputVideo;
	inputVideo.open(0);
	cv::Mat img;
	inputVideo.read(img);

	// 检测器声明
	marker::MarkerDetector marker_detector(img.size());
	// 用来接收检测到的单个marker信息
	marker::MarkerInfo marker_info;
	// 用来接收检测到的多个marker信息
	std::vector<marker::MarkerInfo> marker_infos;
	do
    	{
        	inputVideo.read(img);
		if (MODE == SINGLE)
		{
			// 单marker检测,即只返回所有marker的第一个
			// 考虑到比赛只用到1个maker,用这个比较方便
			// 参数说明: (输入图像, 可视化, 版本选择, 模式选择)
			// 输入图像: 
			// 可视化: 是否在图像中为marker画出坐标轴和中心点
			// 版本选择: VER_OPENCV or VER_ARUCO
			// 模式选择: MODE_DETECT or MODE_TRACK
			// VER_OPENCV目前只有DETECT模式
			marker_info = marker_detector.detect_single_marker(img, true, marker::VER_OPENCV, marker::MODE_DETECT);
			// 通过id判断是否检测到了marker,没有marker会返回id为-1的marker_info
			if (marker_info.id >= 0)
			{
				show_marker_info(marker_info);
			}else
			{
				std::cout << "未检测到marker" << std::endl;
			}
		}
		else
		{
			// 全部marker检测,即返回所有marker
			// 参数含义同上
			marker_infos = marker_detector.detect_markers(img, true, marker::VER_OPENCV, marker::MODE_DETECT);
			// 通过vector中是否有元素来判断检测结果
			if (marker_infos.size() > 0)
			{
				for (auto m:marker_infos)
				{
					show_marker_info(m);
				}
			}else
			{
				std::cout << "未检测到marker" << std::endl;
			}
		}
		
		cv::imshow("in", img);
    	} while (char(cv::waitKey(20)) != 27 && inputVideo.grab());  // wait for esc to be pressed

	return 0;
}

```
