# ResDet

## TODO

1. 解决启用UART时输出不只换行不回车问题
2. 屏蔽**invalid ue golomb code**错误

## 需要注意的问题

- 抓取范围半径约10cm
- 抓手抓取一次大致耗时18s
- 深度传感器以开启时压强校零, 因此要在下水前开机才能测得水的绝对高度.
- 深度传感器记录的是压强, 当工作水域密度和传感器内置密度不同时传感器给出的深度不准

   解决办法: 现场测量定量水的重量得出修正参数.

   $$\rho_{sea}g_{local}V=k\rho_0g_0V$$
   $$k=\frac{G_{sea}}{G_{standard}}$$
   $$h=\frac{h}{k}'$$

🔗[海水密度参考资料](doc/海水密度.png)

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
