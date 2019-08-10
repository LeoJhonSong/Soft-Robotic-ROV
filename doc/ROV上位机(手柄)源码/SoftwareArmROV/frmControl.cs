using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.Drawing.Drawing2D;
using System.Threading;
using System.Net.Sockets;
using System.Net;


namespace SoftwareArmROV
{
    public partial class frmControl : Form
    {
        #region 变量
        private static Socket socketSend; //发送数据
        private static Socket socketWatch;//通信
        private Joystick joy; //手柄处理
        private JosystickSend sends; //发送数据对象

        private int isConModel = 1;    //控制模式（闭环）
        private int isOpeType = 1;    //操作类型（手柄操作）
        private bool isAround;      //前后触发标识
        private bool isRotate;      //旋转触发标识
        private bool isTranslation; //平移触发标识
        #endregion


        public frmControl()
        {
            InitializeComponent();
            TextBox.CheckForIllegalCrossThreadCalls = false;//关闭跨线程修改控件检查
        }


        /// <summary>
        /// 画面初期化
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void frmControl_Load(object sender, EventArgs e)
        {
            //初始化画面滚动条及滑动条
            init();

            //注册事件(画面皮肤)
            this.skE.SkinFile = "SteelBlack.ssk";

            //设置画面报警标识
            setShape(picControlModule, Color.GreenYellow);
            setShape(picMainDeckCabin, Color.GreenYellow);
        }

        #region 初始化init
        /// <summary>
        /// 初始化画面滚动条数据
        /// </summary>
        private void init()
        {
            //摇杆状态显示
            hscX.Value = 128;//左移/右移
            vscY.Value = 128;//上浮/下潜
            hscZ.Value = 128;//左转/右转
            vscR.Value = 128;//前进/后退 

            //加载操作类型
            List<string> list = new List<string>();
            list.Add("手柄操作");
            list.Add("UI操作");
            list.Add("PID参数");
            cmbHandle.DataSource = list;

            //舵机操作
            trbSteering_Engine1.Value = 1000;//舵机1
            trbSteering_Engine2.Value = 1000;//舵机2

            //电机测试
            setMotorTest();
            
        }

        /// <summary>
        /// 设置电机测试
        /// </summary>
        private void setMotorTest()
        {
            trbVertical.Value = 128;//垂直
            trbAround.Value = 128;//前后
            trbRotate.Value = 128;//旋转
            trbTranslation.Value = 128;//平移
        }

        /// <summary>
        /// 设置画面报警状态
        /// </summary>
        private void setShape(PictureBox pic, Color color)
        {
            //构建图像图形（修整为圆形）
            GraphicsPath gp = new GraphicsPath();
            gp.AddEllipse(pic.ClientRectangle);
            Region region = new Region(gp);
            pic.Region = region;
            gp.Dispose();
            region.Dispose();
            pic.BackColor = color;//设置画面控件颜色
        }

        /// <summary>
        /// 显示系统时间
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void tmrSys_Tick(object sender, EventArgs e)
        {
            DateTime dt = DateTime.Now;
            this.tssTime.Text = "系统当前时间：" + string.Format("{0:G}", dt);
        }
        #endregion

        #region 用户连接设置
        /// <summary>
        /// 连接设置
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void btnConSocket_Click(object sender, EventArgs e)
        {
            try
            {
                if (this.btnConSocket.Text == "创建服务器")
                {
                    //点击开始监听时 在服务端创建一个负责监听IP和端口号的Socket
                    socketWatch = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
                    //创建 ip对象
                    IPAddress address = IPAddress.Parse(this.txtIP.Text.Trim());
                    //创建对象端口
                    IPEndPoint point = new IPEndPoint(address, Convert.ToInt32(this.txtPort.Text.Trim()));
                    socketWatch.Bind(point);//绑定端口号
                    socketWatch.Listen(10);//设置监听

                    //创建监听线程
                    Thread thread = new Thread(Listen);
                    thread.IsBackground = true;
                    thread.Start(socketWatch);
                    this.grpOperation.Enabled = true;

                    toolSStLbl.Text = "服务器创建成功   ";
                    this.btnConSocket.Text = "断开服务器";
                }
                else
                {
                    socketWatch.Close();
                    socketSend.Close();
                    this.btnConSocket.Text = "创建服务器";
                    toolSStLbl.Text = "服务器连接已断开   ";
                }
                
            }
            catch (Exception ex)
            {
                toolSStLbl.Text = "服务器异常   ";
                this.grpOperation.Enabled = false;
            }
        }
        /// <summary>
        /// 等待客户端的连接 并且创建与之通信的Socket
        /// </summary>

        void Listen(object o)
        {
            try
            {
                Socket socketWatch = o as Socket;
                while (true)
                {
                    socketSend = socketWatch.Accept();//等待接收客户端连接
                    //开启一个新线程，执行接收消息方法
                    Thread r_thread = new Thread(NetReceiveData);
                    r_thread.IsBackground = true;
                    r_thread.Start(socketSend);
                }
            }
            catch { }
        }
        /// <summary>
        /// 服务器端不停的接收客户端发来的消息
        /// </summary>
        /// <param name="o"></param>
        void NetReceiveData(object o)
        {
            try
            {
                Socket socketReceive = o as Socket;
                while (true)
                {
                    //客户端连接服务器成功后，服务器接收客户端发送的消息
                    byte[] data = new byte[1024 * 1024 * 3];
                    //实际接收到的有效字节数
                    int len = socketReceive.Receive(data);
                    if (len == 0)
                    {
                        break;
                    }
                    ////接收客户端发送数据
                    if (len >= 28)
                    {
                        //获取通信报文
                        for (int i = 0; i < len; i++)
                        {
                            //获取通信报文
                            if ((data[0] == 0xFE) && (data[1] == 0xFE) && (data[26] == 0xFD) && (data[27] == 0xFD))
                            {
                                byte sum = data[0];
                                for (int j = 1; j <= len - 4; j++)
                                {
                                    sum = Convert.ToByte(sum ^ data[j]);
                                }
                                if (sum == data[25])
                                {

                                    //控制舱温度
                                    this.lblTemperature.Text = ((float)((data[2] * 256 + data[3])) / 10).ToString("0.00");

                                    //控制舱漏水报警
                                    if (data[4] == 0xAA)
                                    {
                                        picControlModule.BackColor = Color.Red;
                                    }
                                    else
                                    {
                                        picControlModule.BackColor = Color.GreenYellow;
                                    }

                                    //主板舱温度
                                    this.lblBoard.Text = (((float)(data[5] * 256 + data[6])) / 10).ToString("0.00");

                                    //主板舱漏水报警
                                    if (data[7] == 0xAA)
                                    {
                                        picMainDeckCabin.BackColor = Color.Red;
                                    }
                                    else
                                    {
                                        picMainDeckCabin.BackColor = Color.GreenYellow;
                                    }

                                    //机器人深度
                                    float fDepth = (float)((data[8] * 256 + data[9])) / 100;
                                    this.lblDepth.Text = fDepth.ToString("0.00");

                                    //左路电压
                                    float fVol = (float)data[10] / 10;
                                    this.lblVoltage1.Text = fVol.ToString("0.00");

                                    //左路电流
                                    float fEle = (float)((data[11] * 256 + data[12])) / 100;
                                    this.lblElectricity1.Text = fEle.ToString("0.00");

                                    //右路电压值
                                    float fVol2 = (float)data[13] / 10;
                                    this.lblVoltage2.Text = fVol2.ToString("0.00");

                                    //右路电流
                                    float fEle2 = (float)((data[14] * 256 + data[15])) / 100;
                                    this.lblElectricity2.Text = fEle2.ToString("0.00");

                                    float dtemp_BoatAngle = GetAttitudeData(data, 3, 22);//航向角
                                    //if (dtemp_BoatAngle > 180)
                                    //{
                                    //    dtemp_BoatAngle = dtemp_BoatAngle - 360;
                                    //}
                                    this.hiIndicator.CurrentHeading = Convert.ToDecimal(dtemp_BoatAngle);
                                }
                            }
                        }
                    }
                }
            }
            catch { }
        }
        /// <summary>
        /// 服务器向客户端发送消息
        /// </summary>
        /// <param name="str"></param>
        void NetSendData(byte[] data)
        {
            try
            {
                socketSend.Send(data);
                this.tssConTerminal.Text = "通信成功   ";
            }
            catch (Exception ex)
            {
                this.tssConTerminal.Text = "通信失败   ";
            }
        }
        private float GetAttitudeData(byte[] body, int length, int index)
        {
            float signedData = 0;

            int temp_value1 = 0;
            int temp_value2 = 0;
            int temp_value3 = 0;
            int temp_value4 = 0;
            int temp_value5 = 0;
            int temp_value6 = 0;

            temp_value1 = body[index] / 16;
            temp_value2 = body[index] % 16;

            temp_value3 = body[index + 1] / 16;
            temp_value4 = body[index + 1] % 16;
            if (length == 3)
            {
                temp_value5 = body[index + 2] / 16;
                temp_value6 = body[index + 2] % 16;
            }

            if (temp_value1 == 1)
            {
                signedData = (float)((-1) * ((temp_value2 * 100) + (temp_value3 * 10) + temp_value4 + ((temp_value5 * 10 + temp_value6) / 100.0)));
            }
            else
            {
                signedData = (float)(((temp_value2 * 100) + (temp_value3 * 10) + temp_value4 + ((temp_value5 * 10 + temp_value6) / 100.0)));
            }

            return signedData;
        }


        /// <summary>
        /// 手柄连接
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void btnConHandle_Click(object sender, EventArgs e)
        {
            try
            {
                string type = this.btnConHandle.Text;
                if (type == "连 接")
                {
                    //如果选择项目为手柄操作
                    if (this.cmbHandle.Text == "手柄操作")
                    {
                        joy = new Joystick(JoystickAPI.JOYSTICKID1);
                        joy.Capture();
                        if (joy.IsCapture == false)
                        {
                            this.tssJoyCon.Text = "手柄未找到   ";
                            return;
                        }
                        tmrHandle.Enabled = true;
                        this.tssJoyCon.Text = "手柄连接成功   ";
                    }
                    this.btnConHandle.Text = "重 置";
                    this.cmbHandle.Enabled = false;
                }
                else
                {
                    tmrHandle.Enabled = false;

                    this.btnConHandle.Text = "连 接";
                    this.cmbHandle.Enabled = true;
                }
            }
            catch (Exception)
            {
                this.tssJoyCon.Text = "手柄连接异常，请检查接口   ";
                throw;
            }
        }

        /// <summary>
        /// 手柄操作定时器
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void tmrHandle_Tick(object sender, EventArgs e)
        {
            //LED及舵角操作
            int iLed = this.trbLED.Value;//LED
            int iSteering_Engine1 = trbSteering_Engine1.Value + 500;//舵机1
            int iSteering_Engine2 = trbSteering_Engine2.Value + 500;//舵机2

            //手柄操作
            JoystickHandle handle = joy.OnTimerCallback();
            vscY.Value = Convert.ToInt32(handle.Ypos / 256);//上浮/下潜
            hscX.Value = Convert.ToInt32(handle.Xpos / 256);//左移/右移
            vscR.Value = Convert.ToInt32(handle.Rpos / 256);//前进后退
            hscZ.Value = Convert.ToInt32(handle.Zpos / 256);//左转/右转

            sends = new JosystickSend();
            sends.UpDown_Value_Y = vscY.Value;//垂直

            //左摇杆X轴数据处理
            if (((handle.Rpos > 32757) && (handle.Rpos < 32777)) && ((handle.Zpos > 32757) && (handle.Zpos < 32777)))//判断右摇杆是否处于中间状态
            {
                sends.Side_Shift_Value_X = hscX.Value;//平移赋值
                sends.Forward_Back_Value_R = 128;//R轴赋中值
                sends.Direc_Value_Z = 128;//Z轴赋中值
            }
            else //右摇杆操作或者左摇杆与右摇杆同时操作
            {
                if ((handle.Zpos >= 32767) && (handle.Rpos <= 32767)) //第一象限
                {
                    if (handle.Rpos <= (65535 - handle.Zpos)) //靠近R周
                    {
                        sends.Forward_Back_Value_R = vscR.Value;//向前走
                        sends.Direc_Value_Z = 128;//Z轴赋中值
                    }
                    else//靠近Z轴
                    {
                        sends.Direc_Value_Z = hscZ.Value;//旋转
                        sends.Forward_Back_Value_R = 128;//R轴赋中值
                    }
                }
                else if ((handle.Zpos < 32767) && (handle.Rpos <= 32767))  //第二象限
                {
                    if (handle.Rpos <= handle.Zpos) //靠近R轴
                    {
                        sends.Forward_Back_Value_R = vscR.Value;
                        sends.Direc_Value_Z = 128;//Z轴赋中值
                    }
                    else //靠近Z轴
                    {
                        sends.Direc_Value_Z = hscZ.Value;//旋转
                        sends.Forward_Back_Value_R = 128;//R轴赋中值
                    }
                }

                else if ((handle.Zpos < 32767) && (handle.Rpos > 32767)) //第三象限
                {
                    if (handle.Rpos < 65535 - handle.Zpos) //靠近Z周
                    {
                        sends.Direc_Value_Z = hscZ.Value;
                        sends.Forward_Back_Value_R = 128;//R轴赋中值
                    }
                    else //靠近R轴
                    {
                        sends.Forward_Back_Value_R = vscR.Value;
                        sends.Direc_Value_Z = 128;
                    }
                }
                else if ((handle.Zpos >= 32767) && (handle.Rpos > 32767)) //第四象限
                {
                    if (handle.Rpos >= handle.Zpos) //靠近R轴
                    {
                        sends.Forward_Back_Value_R = vscR.Value;
                        sends.Direc_Value_Z = 128;//Z轴赋中值
                    }
                    else //靠近Z轴
                    {
                        sends.Direc_Value_Z = hscZ.Value;
                        sends.Forward_Back_Value_R = 128;//R轴赋中值
                    }
                }

                sends.Side_Shift_Value_X = 128;//平移中值
            }

            byte[] sendDate = SendTransmitData(iLed, iSteering_Engine1, iSteering_Engine2, sends);

            NetSendData(sendDate);//下发手柄操作数据
        }

        /// <summary>
        /// UI操作下发数据
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void tmrUI_Tick(object sender, EventArgs e)
        {
            //LED及舵角操作
            int iLed = this.trbLED.Value;//LED
            int iSteering_Engine1 = trbSteering_Engine1.Value + 500;//舵机1
            int iSteering_Engine2 = trbSteering_Engine2.Value + 500;//舵机2

            //点击测试
            sends = new JosystickSend();
            sends.UpDown_Value_Y = trbVertical.Value;//垂直
            sends.Forward_Back_Value_R = trbAround.Value;//前后
            sends.Direc_Value_Z = trbRotate.Value;//旋转
            sends.Side_Shift_Value_X = trbTranslation.Value;//平移

            if (isAround)//前后
            {
                sends.Forward_Back_Value_R = trbAround.Value;//前后
                sends.Side_Shift_Value_X = 128;//平移
                sends.Direc_Value_Z = 128;//旋转
            }
            if (isRotate)//旋转
            {
                sends.Direc_Value_Z = trbRotate.Value;//旋转
                sends.Forward_Back_Value_R = 128;//前后
                sends.Side_Shift_Value_X = 128;//平移
            }
            if (isTranslation)//平移
            {
                sends.Side_Shift_Value_X = trbTranslation.Value;//平移
                sends.Forward_Back_Value_R = 128;//前后
                sends.Direc_Value_Z = 128;//旋转  
            }

            byte[] sendDate = SendTransmitData(iLed, iSteering_Engine1, iSteering_Engine2, sends);

            NetSendData(sendDate);//下发UI操作数据
        }

        /// <summary>
        /// 下发数据组
        /// </summary>
        /// <param name="iLed">LED</param>
        /// <param name="iSteering_Engine1">舵机1</param>
        /// <param name="iSteering_Engine2">舵机2</param>
        /// <param name="SendMotorData">手柄操作（手柄）/UI操作（测试）</param>
        /// <returns></returns>
        public byte[] SendTransmitData(int iLed, int iSteering_Engine1, int iSteering_Engine2, JosystickSend SendMotorData)
        {

            byte[] theData = new byte[27];
            int inforData = GetInfoWord();

            int index = 0;
            //报文头
            theData[index++] = 0xfe;
            theData[index++] = 0xfe;

            theData[index++] = Convert.ToByte(inforData / 256);
            theData[index++] = Convert.ToByte(inforData % 256); ;

            //LED
            theData[index++] = Convert.ToByte(iLed / 256);//高位
            theData[index++] = Convert.ToByte(iLed % 256);//低位
            theData[index++] = Convert.ToByte(iLed / 256);//高位
            theData[index++] = Convert.ToByte(iLed % 256);//低位

            //舵机1
            theData[index++] = Convert.ToByte(iSteering_Engine1 / 256);//高位
            theData[index++] = Convert.ToByte(iSteering_Engine1 % 256);//低位

            //舵机2
            theData[index++] = Convert.ToByte(iSteering_Engine2 / 256);//高位
            theData[index++] = Convert.ToByte(iSteering_Engine2 % 256);//低位

            //空值
            for (int i = 0; i < 8; i++)
            {
                theData[index++] = 0x00;
            }

            //运动控制数据
            theData[index++] = Convert.ToByte(SendMotorData.Forward_Back_Value_R);  //电机前后运动控制值
            theData[index++] = Convert.ToByte(SendMotorData.Side_Shift_Value_X);    //电机平移运动控制值
            theData[index++] = Convert.ToByte(SendMotorData.Direc_Value_Z);         //电机转向运动控制值
            theData[index++] = Convert.ToByte(SendMotorData.UpDown_Value_Y);        //电机垂直运动控制值

            byte sum = theData[0];
            for (int i = 1; i <= index - 1; i++)
            {
                sum = Convert.ToByte(sum ^ theData[i]);
            }
            //校验和
            theData[index++] = sum;

            //结尾
            theData[index++] = 0xfd;
            theData[index++] = 0xfd;
            return theData;
        }

        /// <summary>
        /// 封装信息字
        /// </summary>
        /// <returns></returns>
        public int GetInfoWord()
        {
            int infor_Data_MostBit = 0;//信息字高位
            int infor_Data_LowBit = 15;//信息字低位
            int infor_Data = 0;//信息字最终数据
            infor_Data_MostBit = isConModel * 2 + isOpeType;
            infor_Data = infor_Data_MostBit * 256 + infor_Data_LowBit;
            return infor_Data;
        }

        #endregion

        #region LED及舵机操作（关闭操作）
        /// <summary>
        /// LED关闭设置
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void btnCloseLED_Click(object sender, EventArgs e)
        {
            trbLED.Value = 0;
        }


        /// <summary>
        /// 舵机1关闭
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void btnClose_Steering_Engine1_Click(object sender, EventArgs e)
        {
            trbSteering_Engine1.Value = 1000;
        }

        /// <summary>
        /// 舵机2关闭
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void btnClose_Steering_Engine2_Click(object sender, EventArgs e)
        {
            trbSteering_Engine2.Value = 1000;
        }

        #endregion

        #region 测试点击设置（关闭操作）
        /// <summary>
        /// 垂直设置
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void btnCloseVertical_Click(object sender, EventArgs e)
        {
            trbVertical.Value = 128;
        }

        /// <summary>
        /// 前后设置
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void btnCloseAround_Click(object sender, EventArgs e)
        {
            trbAround.Value = 128;
        }

        /// <summary>
        /// 旋转设置
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void btnCloseRotate_Click(object sender, EventArgs e)
        {
            trbRotate.Value = 128;
        }

        /// <summary>
        /// 平移设置
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void btnCloseTranslation_Click(object sender, EventArgs e)
        {
            trbTranslation.Value = 128;
        }
        #endregion

        #region 电机测试设置操作标识

        //设置测试标识
        private void SetTrbScroll()
        {
            isAround = false;
            isRotate = false;
            isTranslation = false;
        }

        /// <summary>
        /// 测试滑动平移划条
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void trbTranslation_Scroll(object sender, EventArgs e)
        {
            SetTrbScroll();
            isTranslation = true;//设置标识
        }

        /// <summary>
        /// 测试滑动前后划条
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void trbAround_Scroll(object sender, EventArgs e)
        {
            SetTrbScroll();
            isAround = true;
        }

        /// <summary>
        /// 测试滑动旋转划条
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void trbRotate_Scroll(object sender, EventArgs e)
        {
            SetTrbScroll();
            isRotate = true;
        }
        #endregion

        #region 操作类型
        /// <summary>
        /// 操作类型
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void cmbHandle_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (cmbHandle.Text.ToString() == "UI操作")
            {
                isOpeType = 0;
                this.btnConHandle.Enabled = false;

                this.trbVertical.Enabled = true;//垂直滚动条可用
                this.trbAround.Enabled = true;//前后滚动条可用
                this.trbRotate.Enabled = true;//旋转滚动条可用
                this.trbTranslation.Enabled = true;//平移滚动条可用

                this.btnCloseVertical.Enabled = true;//垂直关闭按钮可用
                this.btnCloseAround.Enabled = true;//前后关闭按钮可用
                this.btnCloseRotate.Enabled = true;//旋转关闭按钮可用
                this.btnCloseTranslation.Enabled = true;//平移关闭按钮可用
                tmrUI.Enabled = true;//启动UI定时器
                this.tmrHandle.Enabled = false;//停用摇杆定时器
                this.ttrType.Text = "UI操作   ";
                this.tssJoyCon.Text = "";

                setMotorTest();//设置电机各项
            }
            else if (cmbHandle.Text.ToString() == "手柄操作")
            {
                isOpeType = 1;
                tmrUI.Enabled = false;
                this.btnConHandle.Enabled = true;//手柄控制按钮可用

                this.trbVertical.Enabled = false;//垂直滚动条不可用
                this.trbAround.Enabled = false;//前后滚动条不可用
                this.trbRotate.Enabled = false;//旋转滚动条不可用
                this.trbTranslation.Enabled = false;//平移滚动条不可用

                this.btnCloseVertical.Enabled = false;//垂直关闭按钮不可用
                this.btnCloseAround.Enabled = false;//前后关闭按钮不可用
                this.btnCloseRotate.Enabled = false;//旋转关闭按钮不可用
                this.btnCloseTranslation.Enabled = false;//平移关闭按钮不可用
                this.ttrType.Text = "手柄操作   ";
            }
            else if (cmbHandle.Text.ToString() == "PID参数")
            {
                //参数设置画面展示
                frmPID f = new frmPID();
                f.Pid += new SendPidParameterInfo(SendPidParameter);
                f.ShowDialog();
            }
        }

        /// <summary>
        /// 下发PID数据
        /// </summary>
        /// <param name="SendType">下发类型（深度、前进、后退、左移、右移）</param>
        /// <param name="SendP">P值</param>
        /// <param name="SendI">I值</param>
        /// <param name="SendD">D值</param>
        private void SendPidParameter(byte SendType, float SendP, float SendI, float SendD, byte SendFlash,string Type)
        {
            byte[] theData = new byte[13];
            int index = 0;
            //报文头
            theData[index++] = 0xFB;
            theData[index++] = 0xFB;

            if (Type=="PID")
            {
                theData[index++] = SendType;
            }
            else if (Type=="FLASH")
            {
                theData[index++] = SendFlash;
            }


            //P值
            theData[index++] = 0x00;
            theData[index++] = Convert.ToByte(Math.Floor(SendP));
            theData[index++] = Convert.ToByte((SendP*100)%100);

            //I值
            theData[index++] = Convert.ToByte(Math.Floor(SendI));
            theData[index++] = Convert.ToByte((SendI * 100) % 100);

            //D值
            theData[index++] = Convert.ToByte(Math.Floor(SendD));
            theData[index++] = Convert.ToByte((SendD * 100) % 100);

            byte sum = theData[0];
            for (int i = 1; i <= index - 1; i++)
            {
                sum = Convert.ToByte(sum ^ theData[i]);
            }
            //校验和
            theData[index++] = sum;

            //结尾
            theData[index++] = 0xFC;
            theData[index++] = 0xFC;

            NetSendData(theData);//下发PID数据
        }
        #endregion

        #region 控制模式设置
        /// <summary>
        /// 设置控制模式为开环
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void rdoOpenLoop_Click(object sender, EventArgs e)
        {
            isConModel = 0;
        }

        /// <summary>
        /// 控制模式为闭环
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void rdoCloseLoop_Click(object sender, EventArgs e)
        {
            isConModel = 1;
        }
        #endregion
    }
}