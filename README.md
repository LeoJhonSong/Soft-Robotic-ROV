# Soft-Robotic-ROV

1. [需要注意的问题](#需要注意的问题)
   1. [有关抓取](#有关抓取)
   2. [有关深度传感器](#有关深度传感器)
   3. [有关TCP通讯](#有关TCP通讯)
   4. [有关识别](#有关识别)
   5. [有关ROV移动](#有关ROV移动)
   6. [有关配重](#有关配重)
   7. [有关密封](#有关密封)
   8. [有关线材](#有关线材)
2. [自主抓取](#自主抓取)
   1. [2020目标](#2020目标)
   2. [程序逻辑](#程序逻辑)
   3. [巡航策略](#巡航策略)
3. [程序说明](#程序说明)
   1. [环境配置](#环境配置)
      1. [libtorch安装](#libtorch安装)
      2. [CUDA安装](#CUDA安装)
      3. [cuDNN安装](#cuDNN安装)
      4. [OpenCV安装](#OpenCV安装)
   2. [识别模型](#识别模型)
      1. [识别类别](#识别类别)
   3. [程序参数](#程序参数)
   4. [程序状态-键盘按键-手柄按键对照表](#程序状态-键盘按键-手柄按键对照表)
   5. [与其他设备通信](#与其他设备通信)
      1. [UART](#UART)
      2. [游戏手柄](#游戏手柄)
      3. [TCP_Server类](#TCP_Server类)
         1. [成员函数](#成员函数)
         2. [sendMsg()中move值与动作与可用宏定义对应表](#sendMsg中move值与动作与可用宏定义对应表)
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

- ROV开启后TCP通信只能连接**一次**, 如果可以改动ROV程序可以让ROV一定时间内未连接就发送连
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

### 有关线材

- 目前的线又多又有粗有细, 很容易因为拧了相互缠住, 开始前要理好线. 可以考虑用波纹管
- 线的接头处一定要确保牢固

## 自主抓取

### 2020目标

三十分钟内能抓到一个目标就nb

### 程序逻辑

![抓取流程](doc/抓取流程.svg)

### 巡航策略

因为没有任何传感器的数据可以借助, (采取任何路线效果都一样) 采用蛇形走位遍历水域.

👇示意图如下. 蓝色为摄像头视野, 橙黄色为路线.

估计巡航时摄像头视野为 `0.6m x 1m`, ROV半速向前约 `0.05m/s`, 半速侧移约
`0.06m/s`, 规定半速向前5s, 半速右移10s, 半速向前5s, 半速左移10s, 半速向前5s, 半
速左移10s, 半速向前5s, 半速右移10s为一个周期. **理想状态**路线纵向步长为 `0.25m`, 横
向步长为 `0.6m`.

💡 为了应对**水流过大**的情况, 可以通过修改`src/rov.cpp`中以下两变量的值来增大巡航的横向和纵向步距, 减少扫描区域重叠.

```Cpp
// in rov.cpp
    int side_sec = 3;
    int for_sec = 3;
```

![巡航路线示意图](doc/巡航路线示意图.png)

## 程序说明

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

#### libtorch安装

将下载下来的压缩包解压出的**libtorch**文件夹放到到`~/local`下

#### CUDA安装

```shell
# Ubuntu下: 略
# Manjaro下 (会自动安装gcc7)
yay -S cuda-10.0
sudo ln -s /opt/cuda-10.0 /usr/local/cuda
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

这里假设OpenCV版本为3.4.11, 如果不同, 更换下面代码中版本号

```shell
# 将下载下来的opencv-3.4.11和opencv_contrib-3.4.11解压到同一文件夹下
# 进入opencv-3.4.11, 创建一个build文件夹
mkdir build
# cmake配置. 仔细查看输出信息没有报错了. 还需要一些依赖没写, 跟着报错安就好👍 (期间会下载一些东西, 如果下不动需要在终端翻墙)
cmake ../ \
-D CMAKE_BUILD_TYPE=Releasee \
-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-3.4.11/modules \
-D BUILD_opencv_python2=OFF \
-D CUDA_HOST_COMPILER=/usr/bin/gcc-7 \
-D WITH_CUDA=ON \
-D CUDA_NVCC_FLAGS="-D FORCE_INLINES" \
-D WITH_GTK=ON ..
# 编译并安装
make -j
sudo make install
```

❗️ 如果不是GTK的OpenCV而是Qt的OpenCV的话似乎是无法区分按键大小写的, 其他按键似乎也会有些问题. python安装的OpenCV是Qt的. `utils/key_test.py`可以测试所按按键被OpenCV识别为什么了.


### 识别模型

#### 识别类别

| 类型 | 值   |
| ---- | ---- |
| 背景 | 0    |
| 海参 | 1    |
| 海胆 | 2    |
| 扇贝 | 3    |
| 海星 | 4    |
| 手臂 | 5    |

❗️ 目前训练集没有给海星, 不识别海星.
❗️ 手臂并非由神经网络识别, 是ArUco识别的.

### 程序参数

|参数|含义|可能值|
|-|-|-|
|RUAS|传统水下图像恢复算法模式|**0**: 不使用/**1**: 滤波/**2**: 滤波+直方图滤波|
|K|RUAS滤波参数|**100**|
|R|RUAS滤波参数|**50**|
|NET_PHASE|神经网络模型选择|**0**: 跳过/**1**: netG图像恢复/**2**: netG图像恢复+目标检测/**3**: 检测|
|MODE|视频流来源|**-2**: 从指定网络摄像头读取/**-1**: 从视频读取/**0**: 从电脑摄像头读取|
|SSD_DIM|SSD算法网络维度|**320**/**512**|
|NETG_DIM|netG网络维度|**256**/**320**/**512**|
|TUB||**1**|
|UART|串口模式|**true**: 启用UART串口/**false**: 不启用UART串口|
|WITH_ROV|ROV连接模式|**true**: 连接ROV运行/**false**: 不连接ROV运行|
|TRACK|是否给出目标坐标|**true**/**false**|

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

#### UART

目前程序通过`/dev/ttyUSB0`(第一个被发现的USB串口设备) 以**9600**的波特率进行串口通信. 只有连接了一个串口设备时`/dev/`下才会有这个设备.

❗️ 另一侧也需要以9600的波特率进行通信, 否则收到的会是乱七八糟的东西.

#### 游戏手柄

对于非蓝牙游戏手柄, 可能是`/dev/js0`, `/dev/input/js0`, `/dev/input/js1`等, 如果识别不到可能需要安装驱动, 比如**xboxdrv**. 可以在终端输入`cat /dev/input/js0 | hexdump`, 然后看看操作手柄终端会不会有新输入, 来确认这个设备是不是要配对的游戏手柄. 根据设备路径可能需要改动`utils/js2key`.

从终端运行项目里`utils/joystick`会输出操作游戏手柄对应的摇杆, 按键事件.

#### TCP_Server类

##### 成员函数

|函数名|功能|
|-|-|
|(构造函数)|创建监听socket, 绑定端口并开始监听端口|
|(析构函数)|关闭业务socket, 关闭监听socket|
|`void recvMsg( void )`|如果未建立业务socket则在accept()处阻塞, 直到建立业务socket. 接收ROV发来的24位数据并处理第4位 (舱1是否漏水, 存至 **isOneLeak**), 第7位 (舱2是否漏水, 存至 **isTwoLeak**), 第8, 9位 (深度信息, 存至 **depth**)数据.|
|`void sendMsg( int move )`|按传递的参数**move**对应的动作 (见下表) 发送指令给ROV|

⚠️ accept()在 `recvMsg()` 中意味着必须先执行一次recvMsg()才能和ROV建立连接.

另外值得注意的是`accept()`和`recv()`在未接收到ROV数据时会一直等待, 即阻塞, 直到接收到数据程序才会继续.

##### sendMsg()中move值与动作与可用宏定义对应表

|动作|move的值|宏定义名|
|-|-|-|
|开灯|SEND_LIGHTS_ON|0|
|全速前进|SEND_FORWARD|1|
|全速后退|SEND_BACKWARD|2|
|全速左|SEND_LEFT|3|
|全速右|SEND_RIGHT|4|
|全速左转|SEND_TURN_LEFT|5|
|全速右转|SEND_TURN_RIGHT|6|
|全速上升|SEND_UP|7|
|全速下降|SEND_DOWN|8|
|半速前进|SEND_HALF_FORWARD|9|
|半速后退|SEND_HALF_BACKWARD|10|
|半速左|SEND_HALF_LEFT|11|
|半速右|SEND_HALF_RIGHT|12|
|半速左转|SEND_HALF_TURN_LEFT|13|
|半速右转|SEND_HALF_TURN_RIGHT|14|
|半速上升|SEND_HALF_UP|15|
|半速下降|SEND_HALF_DOWN|16|
|悬停|SEND_SLEEP|17|
|全速坐底并向前微调|SEND_DIVE_FORWARD|18|
|全速坐底并向后微调|SEND_DIVE_BACKWARD|19|
|全速坐底并向左微调|SEND_DIVE_LEFT|20|
|全速坐底并向右微调|SEND_DIVE_RIGHT|21|
|全速坐底并微左转|SEND_DIVE_TURN_LEFT|22|
|全速坐底并微右转|SEND_DIVE_TURN_RIGHT|23|
|向前微调|SEND_ADJUST_FORWARD|24|
|向后微调|SEND_ADJUST_BACKWARD|25|
|向左微调|SEND_ADJUST_LEFT|26|
|向右微调|SEND_ADJUST_RIGHT|27|
|微左转|SEND_ADJUST_TURN_LEFT|28|
|微右转|SEND_ADJUST_TURN_RIGHT|29|

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