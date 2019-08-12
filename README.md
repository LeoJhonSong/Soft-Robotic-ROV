# ResDet

1. [TODO](#todo)
2. [需要注意的问题](#需要注意的问题)
3. [自主抓取](#自主抓取)
   1. [2019目标](#2019目标)
   2. [程序逻辑](#程序逻辑)
   3. [水域分析](#水域分析)
4. [TCP_Server类](#tcp_server类)
   1. [成员函数](#成员函数)
   2. [sendMsg()中move值与动作与可用宏定义对应表](#sendmsg中move值与动作与可用宏定义对应表)

## TODO

1. 解决启用UART时输出不只换行不回车问题
2. 屏蔽**invalid ue golomb code**错误

## 需要注意的问题

- 抓取范围半径约**10cm**
- 抓手抓取一次大致耗时**18s**, 但可以更快
- 深度传感器以**ROV开启时压强**校零, 因此要在**下水前开机**才能测得水的绝对高度,
  即准确的压强补偿
- 深度传感器记录的是压强, 当工作水域密度和传感器内置密度不同时传感器给出的深度不准

   解决办法: 现场测量定量水的重量得出修正参数`k`.

   $$\rho_{sea}g_{local}V=k\rho_0g_0V$$
   $$P_{sensor}=\rho_{sea}g_{local}h_{real}=k\rho_0g_0h_{real}=\rho_0g_0h_{inaccurate}$$
   $$k=\frac{G_{sea}}{G_{standard}}$$
   $$h_{real}=\frac{h_{inaccurate}}{k}$$

   📑 [海水密度参考资料](doc/海水密度.png)  
   📑 [高精度深度传感器参考资料](doc/基于MEMS微系统的深度计系统构建及精度控制.pdf)

- 移动时深度传感器数据变化剧烈, 可以考虑制作如下整流罩 (灰色), 能极大减小水流干扰

![深度传感器整流罩示意图](doc/深度传感器整流罩示意图.jpg)

- ROV开启后TCP通信只能连接一次
- 通过TCP和ROV通信只发一次指令似乎可能收不到, 保险一点最好多次发送
- 软体臂在ROV转动时易被甩起干扰摄像头视野, 甚至可能被识别为目标. 不过在平动时甩动较小
- 因为没有加速度传感器, 无法保证水平方向的完全静止, 如果不在下潜过程中实时调整水
  平坐标始终无法保证坐底后目标仍在可抓范围内. 不过我认为这不是大问题, ROV的误差
  应当由软体臂的抓取范围来容错.

## 自主抓取

### 2019目标

三十分钟内能抓到一个目标就nb

### 程序逻辑

![抓取流程](抓取流程.svg)

### 水域分析

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

## TCP_Server类

### 成员函数

|函数名|功能|
|-|-|
|(构造函数)|创建监听socket, 绑定端口并开始监听端口|
|(析构函数)|关闭业务socket, 关闭监听socket|
|**void recvMsg( void )**|如果未建立业务socket则在accept()处阻塞, 直到建立业务socket. 接收ROV发来的24位数据并处理第4位 (舱1是否漏水, 存至 `isOneLeak`), 第7位 (舱2是否漏水, 存至 `isTwoLeak`), 第8, 9位 (深度信息, 存至 `depth`)数据.|
|**void sendMsg( int move )**|按传递的参数`move`对应的动作 (见下表) 发送指令给ROV|

⚠️ accept()在 **recvMsg()** 中意味着必须先执行一次recvMsg()才能和ROV建立连接.

另外值得注意的是**accept()**和**recv()**在未接收到ROV数据时会一直等待, 即阻塞, 直到接收到数据程序才会继续.

### sendMsg()中move值与动作与可用宏定义对应表

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

