/* ***********************************************
 * Author		:  kingthy
 * Email		:  kingthy@gmail.com
 * DateTime		:  2009-3-27 22:43:36
 * Description	:  游戏手柄的封装类
 *
 * ***********************************************/

using System;
using System.Collections.Generic;
using System.Text;
using System.Runtime.InteropServices;
using System.Threading;
using System.Diagnostics;

namespace SoftwareArmROV
{
    /// <summary>
    /// 游戏手柄类
    /// </summary>
    public class Joystick : IDisposable
    {
        string strResult = string.Empty;
        /// <summary>
        /// 根据游戏手柄的Id实例化
        /// </summary>
        /// <param name="joystickId"></param>
        public Joystick(int joystickId) 
        { 
            this.Id = joystickId;
            this.JoystickCAPS = new JoystickAPI.JOYCAPS();

            //取得游戏手柄的参数信息
            if (JoystickAPI.joyGetDevCaps(joystickId, ref this.JoystickCAPS, Marshal.SizeOf(typeof(JoystickAPI.JOYCAPS)))
                == JoystickAPI.JOYERR_NOERROR)
            {
                this.IsConnected = true;
                this.Name = this.JoystickCAPS.szPname;
            }
            else
            {
                this.IsConnected = false;
            }

        }

        /// <summary>
        /// 返回当前游戏手柄的Id
        /// </summary>
        public int Id { get; private set; }

        /// <summary>
        /// 返回当前游戏手柄的名称
        /// </summary>
        public string Name { get; private set; }

        /// <summary>
        /// 返回当前游戏手柄是否已连接
        /// </summary>
        public bool IsConnected { get; private set; }

        /// <summary>
        /// 是否已捕捉
        /// </summary>
        public bool IsCapture { get; set; }

        /// <summary>
        /// 游戏手柄的参数信息
        /// </summary>
        private JoystickAPI.JOYCAPS JoystickCAPS;

        /// <summary>
        /// 游戏手柄的参数信息
        /// </summary>
        private JoystickHandle JoyHandle;
        /// <summary>
        /// 定时器
        /// </summary>
        private Timer CaptureTimer;

        #region 事件定义
        /// <summary>
        /// 按钮被单击
        /// </summary>
        public event EventHandler<JoystickEventArgs> Click;
        /// <summary>
        /// 按钮被按下
        /// </summary>
        public event EventHandler<JoystickEventArgs> ButtonDown;
        /// <summary>
        /// 按钮已弹起
        /// </summary>
        public event EventHandler<JoystickEventArgs> ButtonUp;
        /// <summary>
        /// 触发单击事件
        /// </summary>
        /// <param name="e"></param>
        protected void OnClick(JoystickEventArgs e)
        {
            EventHandler<JoystickEventArgs> h = this.Click;
            if (h != null) h(this, e);
        }
        /// <summary>
        /// 触发按钮弹起事件
        /// </summary>
        /// <param name="e"></param>
        protected void OnButtonUp(JoystickEventArgs e)
        {
            EventHandler<JoystickEventArgs> h = this.ButtonUp;
            if (h != null) h(this, e);
        }
        /// <summary>
        /// 触发按钮按下事件
        /// </summary>
        /// <param name="e"></param>
        protected void OnButtonDown(JoystickEventArgs e)
        {
            EventHandler<JoystickEventArgs> h = this.ButtonDown;
            if (h != null) h(this, e);
        }
        #endregion

        /// <summary>
        /// 捕捉游戏手柄
        /// </summary>
        /// <returns></returns>
        public void Capture()
        {
            if (this.IsConnected && !this.IsCapture)
            {
                //手柄已连接
                this.IsCapture = true;
                //this.CaptureTimer = new Timer(this.OnTimerCallback, null, 0, 50);
            }
        }

        /// <summary>
        /// 释放捕捉
        /// </summary>
        public void ReleaseCapture()
        {
            if (this.IsCapture)
            {
                this.CaptureTimer.Dispose();
                this.CaptureTimer = null;
                this.IsCapture = false;
            }
        }

        /// <summary>
        /// 前一次的处于按下状态的按钮
        /// </summary>
        private JoystickButtons PreviousButtons = JoystickButtons.None;

        /// <summary>
        /// 定时器的回调方法
        /// </summary>
        /// <param name="state"></param>
        //private void OnTimerCallback(object state)
        public JoystickHandle OnTimerCallback()
        {
            JoyHandle = new JoystickHandle();
            JoystickAPI.JOYINFOEX infoEx = new JoystickAPI.JOYINFOEX();
            infoEx.dwSize = Marshal.SizeOf(typeof(JoystickAPI.JOYINFOEX));
            infoEx.dwFlags = (int)JoystickAPI.JOY_RETURNBUTTONS;

            int result = JoystickAPI.joyGetPosEx(this.Id, ref infoEx);

            if (result == JoystickAPI.JOYERR_NOERROR)
            {
                
                JoyHandle.Xpos = infoEx.dwXpos;
                JoyHandle.Ypos = infoEx.dwYpos;
                JoyHandle.Zpos = infoEx.dwZpos;
                JoyHandle.Rpos = infoEx.dwRpos;
            }

            return JoyHandle;
        }


        /// <summary>
        /// 根据按钮码获取当前按下的按钮
        /// </summary>
        /// <param name="dwButtons"></param>
        /// <returns></returns>
        private JoystickButtons GetButtons(int dwButtons)
        {
            JoystickButtons buttons = JoystickButtons.None;
            if ((dwButtons & JoystickAPI.JOY_BUTTON1) == JoystickAPI.JOY_BUTTON1)
            {
                buttons |= JoystickButtons.B1;
            }
            if ((dwButtons & JoystickAPI.JOY_BUTTON2) == JoystickAPI.JOY_BUTTON2)
            {
                buttons |= JoystickButtons.B2;
            }
            if ((dwButtons & JoystickAPI.JOY_BUTTON3) == JoystickAPI.JOY_BUTTON3)
            {
                buttons |= JoystickButtons.B3;
            }
            if ((dwButtons & JoystickAPI.JOY_BUTTON4) == JoystickAPI.JOY_BUTTON4)
            {
                buttons |= JoystickButtons.B4;
            }
            if ((dwButtons & JoystickAPI.JOY_BUTTON5) == JoystickAPI.JOY_BUTTON5)
            {
                buttons |= JoystickButtons.B5;
            }
            if ((dwButtons & JoystickAPI.JOY_BUTTON6) == JoystickAPI.JOY_BUTTON6)
            {
                buttons |= JoystickButtons.B6;
            }
            if ((dwButtons & JoystickAPI.JOY_BUTTON7) == JoystickAPI.JOY_BUTTON7)
            {
                buttons |= JoystickButtons.B7;
            }
            if ((dwButtons & JoystickAPI.JOY_BUTTON8) == JoystickAPI.JOY_BUTTON8)
            {
                buttons |= JoystickButtons.B8;
            }
            if ((dwButtons & JoystickAPI.JOY_BUTTON9) == JoystickAPI.JOY_BUTTON9)
            {
                buttons |= JoystickButtons.B9;
            }
            if ((dwButtons & JoystickAPI.JOY_BUTTON10) == JoystickAPI.JOY_BUTTON10)
            {
                buttons |= JoystickButtons.B10;
            }

            return buttons;
        }

        /// <summary>
        /// 获取X,Y轴的状态
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="buttons"></param>
        private void GetXYButtons(int x, int y, ref JoystickButtons buttons)
        {
            //处理X,Y轴
            int m = 0xFFFF / 2;                             //中心点的值,偏差0x100
            if ((x - m) > 0x100)                      
            {
                buttons |= JoystickButtons.Right;
            }
            else if ((m - x) > 0x100)
            {
                buttons |= JoystickButtons.Left;
            }
            if ((y - m) > 0x100)
            {
                buttons |= JoystickButtons.Down;
            }
            else if ((m - y) > 0x100)
            {
                buttons |= JoystickButtons.UP;
            }
        }
        #region IDisposable 成员
        /// <summary>
        /// 释放资源
        /// </summary>
        public void Dispose()
        {
            this.ReleaseCapture();
            this.CaptureTimer = null;
        }

        #endregion
    }
}
