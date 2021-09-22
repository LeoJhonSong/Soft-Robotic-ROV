# Soft-Robotic-ROV

1. [需要注意的问题](#需要注意的问题)
   1. [有关抓取](#有关抓取)
   2. [有关深度传感器](#有关深度传感器)
   3. [有关TCP通讯](#有关tcp通讯)
   4. [有关识别](#有关识别)
   5. [有关ROV移动](#有关rov移动)
   6. [有关配重](#有关配重)
   7. [有关密封](#有关密封)
2. [自主抓取](#自主抓取)
   1. [程序逻辑](#程序逻辑)
   2. [巡航策略](#巡航策略)
3. [程序说明](#程序说明)
   1. [如何启动](#如何启动)
   2. [环境配置](#环境配置)
      1. [libtorch安装](#libtorch安装)
      2. [CUDA安装](#cuda安装)
      3. [cuDNN安装](#cudnn安装)
      4. [OpenCV安装](#opencv安装)
   3. [识别模型](#识别模型)
      1. [识别类别](#识别类别)
   4. [Cpp Visual Server程序参数](#cpp-visual-server程序参数)
      1. [测试](#测试)
   5. [程序状态-键盘按键-手柄按键对照表](#程序状态-键盘按键-手柄按键对照表)
   6. [与其他设备通信](#与其他设备通信)
      1. [游戏手柄](#游戏手柄)
4. [水域分析](#水域分析)

## 需要注意的问题

### 有关抓取

- 抓取范围半径约**10cm**
- 抓手抓取一次耗时将近**60s**
- 2019在**约12米**的海域抓取手臂不能伸到底, 目前原因不明, 好像是手爪飘起来了
- 气管过长会导致气压传输缓慢, 软体臂移动缓慢

### 有关深度传感器

- 深度传感器以**ROV开启时压强**校零, 因此要在**下水前开机**才能测得水的绝对高度,
  即准确的压强补偿
- 深度传感器记录的是**压强**, 当工作水域密度和传感器内置密度不同时传感器给出的深度不准

   解决办法: 现场测量定量水的重量得出修正参数`k`.

   $$\rho_{sea}g_{local}V=k\rho_0g_0V$$
   $$P_{sensor}=\rho_{sea}g_{local}h_{real}=k\rho_0g_0h_{real}=\rho_0g_0h_{inaccurate}$$
   $$k=\frac{G_{sea}}{G_{standard}}$$
   $$h_{real}=\frac{h_{inaccurate}}{k}$$

   📑 [海水密度参考资料](doc/海水密度.png)
   📑 [高精度深度传感器参考资料](doc/基于MEMS微系统的深度计系统构建及精度控制.pdf)

- 原本以为真实水域海底深度波动幅度应该比较大, 结果只有1cm

### 有关TCP通讯

- ROV开启后TCP通信只能连接**一次**, 因为不知道怎么把ROV上的写成socket client了... 如果可以改动ROV程序可以让ROV一定时间内未连接就发送连
  接请求
- 通过TCP和ROV通信只发一次指令似乎可能收不到, 保险一点最好多次发送

### 有关识别

- 软体臂在ROV转动时易被甩起干扰摄像头视野, 甚至手爪, 手臂都可能被识别为目标. 不
  过在平动时甩动较小. 目前以滤掉视野底部的目标来避免此类问题
- 附着到镜头上的灰尘容易被误解为扇贝, 下水前最好擦拭镜头. 涂抹凡士林也能很好地防
  止灰尘附着
- 探灯的光斑也可能被误识别为扇贝.
- 有些石头容易被误解为扇贝
- 定高高度可能需要根据现场能见度调整
- **前半部分上仰**时侧面摄像头容易被阳光直射, 导致视野内只有一片亮光

### 有关ROV移动

- 因为只有角加速度传感器没有加速度传感器, 理论上的悬停也无法实现, 更不用说更多基
  于当前速度的运动, 比如PID
- ROV半速速度和微调时速度尺度需现场调试, 或者可以写进程序让ROV自己在开始时根据移
  动时帧差调整前后, 左右的速度尺度. 要注意参数是**各向异性**的
- ROV在开始前进时会稍微**左偏**一点, 在侧移时是扭动着侧移的, 不过方向确实是那个方向
- 左后和右前电机在静止时容易自己转, 原因不明
- 最好还是给螺旋桨两侧加上滤网, 避免缠上海草, 编织袋丝等
- 在真实海域海底干扰因素太多, 目前的条件不支持使用复杂而**要求精度高**的巡航算
  法, 走一步坐底一下不停抓是目前简陋但最实用的策略
- 感觉可以向潜水员询问一些经验, 因为今天录像那个潜水员看起来既没有铅块也没有缆绳
  但能在水底很稳定
- ROV下水后可能不好**把握方位**, 可以根据ROV的角加速度传感器的值指示出ROV与船的
  相对方位. 既然上位机的UI可以显示ROV角度那就可以用当前角度和初始角度得出ROV和船
  的相对方位

### 有关配重

- 配重一定不能让前半部分较低, 不然抓到网框里的目标很有可能掉出去

### 有关密封

- 密封舱一定要够结实, 不然会爆
- 可以考虑往密封舱里放几个干燥剂
- 手臂和手爪要密封好

## 自主抓取

### 程序逻辑

![抓取流程](doc/抓取流程.svg)

### 巡航策略

因为没有任何传感器的数据可以借助, (采取任何路线效果都一样) 采用蛇形走位遍历水域.

👇示意图如下. 蓝色为摄像头视野, 橙黄色为路线.

估计巡航时摄像头视野为 `0.6m x 1m`, ROV半速向前约 `0.05m/s`, 半速侧移约
`0.06m/s`, 规定半速向前3s, 半速右移3s, 半速向前3s, 半速左移3s, 半速向前3s, 半
速左移3s, 半速向前3s, 半速右移3s为一个周期.
向步长为 `0.6m`.

💡 为了应对**水流过大**的情况, 可以通过修改`src/rov.cpp`中以下两变量的值来增大巡航的横向和纵向步距, 减少扫描区域重叠.

❗️ 由于海底其实不怎么平坦, 加之深度传感器精度不够高, 巡航**两个周期**后坐底一次, 更新巡航高度.

```Cpp
// in rov.cpp
    int side_sec = 3;
    int for_sec = 3;
```

![巡航路线示意图](doc/巡航路线示意图.png)

## 程序说明

### 如何启动

1. 进入项目根目录
2. 启动Visual Server `./build/Soft-Robotics-ROV --stream=camera --bi=true`
3. 启动Control Server `./server/ROV-Control-Server.py`

### 环境配置

| 依赖     | 版本 | 备注                                                         |
| -------- | ---- | ------------------------------------------------------------ |
| libtorch | 1.1  | [下载地址](https://download.pytorch.org/libtorch/cu100/libtorch-shared-with-deps-1.1.0.zip) |
| CUDA     | 10.0 | [下载地址](https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=runfilelocal) (Ubuntu用) |
| cuDNN    |      | 对应CUDA10.0的版本即可. [下载地址](https://developer.nvidia.com/rdp/cudnn-archive) |
| OpenCV   | 3.4  | [opencv下载地址](https://github.com/opencv/opencv/releases) [opencv-contrib下载地址](https://github.com/opencv/opencv_contrib/releases) |
| gcc      | 7    | 编译有CUDA支持的OpenCV用                                     |

❗️ 写明版本的几个依赖不能使用更高的版本, 否则会出错, 详见[#15](https://github.com/leojhonsong/soft-robotic-rov/issues/15)

❗️ 环境配置需要按下面这个顺序来

❗Jetson AGX Xavier上环境配置见[Jetson配置-软件配置](doc/Jetson_AGX_Xavier/Jetson配置.md#软件配置)

#### libtorch安装

将下载下来的压缩包解压出的**libtorch**文件夹放到到`~/local`下

#### CUDA安装

```shell
# Ubuntu下: 略
# Manjaro下 (会自动安装gcc7)
yay -S cuda-10.0
sudo ln -s /opt/cuda-10.0 /usr/local/cuda
sudo ln -s /usr/lib/libstdc++.so /opt/cuda-10.0/lib64/libstdc++.so
# 测试. 应当会输出一串状态信息
cd /usr/local/cuda/samples/1_Utilities/deviceQuery
sudo make
./deviceQuery
```

#### cuDNN安装

```shell
# 进入解压出的cuda文件夹
sudo cp include/cudnn.h /usr/local/cuda/include/
sudo cp lib64/libcudnn* /usr/local/cuda/lib64/
sudo chmod a+r /usr/local/cuda/include/cudnn.h
sudo chmod a+r /usr/local/cuda/lib64/libcudnn*
```

#### OpenCV安装

[OpenCV编译参数官方说明](https://docs.opencv.org/master/db/d05/tutorial_config_reference.html)

这里假设OpenCV版本为3.4.11, 如果不同, 更换下面代码中版本号

```shell
yay -S atlas-lapack eigen
# 将下载下来的opencv-3.4.11和opencv_contrib-3.4.11解压到同一文件夹下
# 进入opencv-3.4.11, 创建一个build文件夹并进入
mkdir build && cd build
# cmake配置. 仔细查看输出信息没有报错了. 还需要一些依赖没写, 跟着报错安就好👍 (期间会下载一些东西, 如果下不动需要在终端翻墙)
cmake \
-D CMAKE_BUILD_TYPE=Release \
-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-3.4.11/modules \
-D BUILD_opencv_python=OFF \
-D BUILD_opencv_python_bindings_generator=OFF \
-D PYTHON2_EXECUTABLE:FILEPATH= \
-D PYTHON3_EXECUTABLE:FILEPATH= \
-D WITH_CUDA=ON \
-D CUDA_NVCC_FLAGS="-D FORCE_INLINES" \
-D OPENCV_DNN_CUDA=ON \
-D ENABLE_FAST_MATH=ON \
-D CUDA_FAST_MATH=ON \
-D WITH_LIBV4L=ON \
-D BUILD_TESTS=OFF \
-D BUILD_PERF_TESTS=OFF \
-D BUILD_EXAMPLES=OFF \
-D BUILD_opencv_apps=OFF \
-D WITH_PROTOBUF=OFF \
-D WITH_GTK=ON \..


-D CUDA_HOST_COMPILER=/opt/cuda-10.0/bin/g++ \
# 编译并安装
make -j 18
sudo make install
```

❗️ 如果不是GTK的OpenCV而是Qt的OpenCV的话似乎是无法区分按键大小写的, 其他按键似乎也会有些问题. python安装的OpenCV是Qt的. `utils/key_test.py`可以测试所按按键被OpenCV识别为什么了.


### 识别模型

#### 识别类别

| 类型     | 值   | 置信值阈值 |
| -------- | ---- | ---------- |
| 背景     | 0    | /          |
| 海参     | 1    | 0.6        |
| 海胆     | 2    | 0.8        |
| 扇贝     | 3    | 0.3        |
| 海星     | 4    | 1.5        |
| 手臂标识 | 5    | /          |

❗️ 目前训练集里没有海星, 因此不识别海星.
❗️ 手臂并非由神经网络识别, 是ArUco识别的.

### Cpp Visual Server程序参数

目标候选项条件为置信度大于阈值`conf_thresh`, 然后每个种类选择置信度最高的一个, 然后按照海参, 海胆, 扇贝的顺序选择. (按比赛中目标得分排序的)

❗即是说在Cpp Visual Server处没有对位置的筛选.

|参数|含义|默认值|可能值|
|-|-|-|-|
|k|RUAS滤波参数|100||
|r|RUAS滤波参数|50||
|ruas|传统水下图像恢复算法选择|0|**0**: 不使用; **1**: 滤波; **2**: 滤波+直方图滤波|
|netg|netG网络维度|256|(见下)|
|ssd|SSD算法网络维度|320|(见下)|
|mode|神经网络模式选择|2|**0**: 跳过; **1**: 仅netG图像恢复; **2**: netG图像恢复+目标检测/**3**: 仅目标检测|
|stream|视频流来源|file|**file**: 从本地文件读取; **link**: 从网址读取; **camera**: 从摄像头读取|
|cid|当从摄像头读取视频时相机id|0|(当不从摄像头读取视频流时该值被忽略)|
|bi|摄像头是否是双目摄像头且只对左路进行识别|false|**true**; **false**|
|address|当从本地文件/网址读取视频时视频地址|./test/test.mp4||
|track|选定单目标跟踪/始终显示所有侯选项|true|**true**; **false**|
|record|是否进行原视频, 已处理视频录制|false|**true**; **false**|
|help|显示帮助信息|||

netG x SSD维度可选组合

| 模式                                 | RUAS | NET_PHASE | SSD维度 | netG维度 |
| ------------------------------------ | -------- | ------- | ------- | ------- |
| 深度学习进行水下图像恢复, 目标检测   | 0     | 2     | 320   | 256  |
| 深度学习进行水下图像恢复, 不检测目标 (2的幂) | 0       | 2       | *        | *       |
| 无水下图像恢复/RUAS恢复, 目标检测 (两维度大小要一样) | 0/2     | 3       | 320/512        | 320/512 |

使用示例
```shell
# 以项目根目录下test/test.mp4作为视频源进行图像恢复及目标识别 (会跳过片头数帧)
./build/Soft-Robotics-ROV
# 以下标为0的双目摄像头左路作为视频源进行图像恢复及目标识别并录像
./build/Soft-Robotics-ROV --stream=camera --bi=true --record=true
# 以网络摄像头为视频源...
./build/Soft-Robotics-ROV --stream=link --address="rtsp://admin:zhifan518@192.168.1.88/11"
```

#### 测试

可以用`echo -e "0\n" | nc 127.0.0.1 8080`或`echo -e "1\n" | nc 127.0.0.1 8080`模拟客户端的一次连接并获得一个json数据.


### 程序状态-键盘按键-手柄按键对照表

|动作|键盘按键|手柄按键|
|-|-|-|
|停止|<kbd>space</kbd>|左肩扳机|
|前进/后退|<kbd>w</kbd>/<kbd>s</kbd>|右摇杆上下|
|左平移/右平移|<kbd>a</kbd>/<kbd>d</kbd>|左摇杆左右|
|前进/后退|<kbd>w</kbd>/<kbd>s</kbd>|右摇杆上下|
|左转/右转|<kbd>A</kbd>/<kbd>D</kbd>|右摇杆左右|
|开灯/关灯|<kbd>L</kbd>/<kbd>l</kbd>||
|坐底, 进入自主控制|<kbd>Enter</kbd>||
|上浮定深|<kbd>u</kbd>||
|巡航|<kbd>c</kbd>||
|坐底至目标处|<kbd>o</kbd>||

❗️ **目前不支持手柄同时有两个动作**. 比如往左上扳左摇杆会往左平移.

### 与其他设备通信

💡 标记和目标的坐标均归一化并乘了100.

![相对位置坐标系](doc/relative_position.png)

#### 游戏手柄

对于非蓝牙游戏手柄, 可能是`/dev/js0`, `/dev/input/js0`, `/dev/input/js1`等, 如果识别不到可能需要安装驱动, 比如**xboxdrv**. 可以在终端输入`cat /dev/input/js0 | hexdump`, 然后看看操作手柄终端会不会有新输入, 来确认这个设备是不是要配对的游戏手柄. 根据设备路径可能需要改动`utils/js2key`.

从终端运行项目里`utils/joystick`会输出操作游戏手柄对应的摇杆, 按键事件.

## 水域分析

所谓自然水域其实就在海洋牧场旁边, 并不怎么自然

![就在海洋牧场旁边](doc/水域参考视频截图/在海洋牧场旁边.png)

水底其实较为平坦, 我看到最大的海沟基本就以下两处

![海沟1](doc/水域参考视频截图/海沟1.png)
![海沟2](doc/水域参考视频截图/海沟2.png)

我感觉扇贝特别多, 海星第二多, 海胆不多, 海参基本没有

👇 海胆, 海参, 海星

![海胆海参](doc/水域参考视频截图/海胆海参.png)
![海星](doc/水域参考视频截图/海星.png)

总的来说基本随便找个地方坐底视野里也有几个扇贝, 有的地方甚至可抓范围内就有好几个
可以抓的扇贝, 不过水底水草较多, 虽然我们假设螺旋桨不会被缠住, 但抓取时很有可能受水草干扰抓不起扇贝, 也许应该设置一个同一目标抓取次数阈值, 抓不住就换一个.

👇 几个坐底时的画面

![坐底1](doc/水域参考视频截图/坐底1.png)
![坐底2](doc/水域参考视频截图/坐底2.png)
![坐底3](doc/水域参考视频截图/坐底3.png)
![坐底4](doc/水域参考视频截图/坐底4.png)