using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.Text.RegularExpressions;

namespace SoftwareArmROV
{
    public delegate void SendPidParameterInfo(byte SendType, float SendP, float SendI, float SendD, byte SendFlash, string Type);
    public partial class frmPID : Form
    {
        public event SendPidParameterInfo Pid;

        public frmPID()
        {
            InitializeComponent();
        }

        /// <summary>
        /// 下发数据
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void btnSend_Click(object sender, EventArgs e)
        {
            //获取画面PID选项
            byte sendType=0x00;
            foreach (RadioButton item in grpOption.Controls)
            {
                if (item.Checked)
                {
                    switch (item.Text)
                    {
                        case "深度":
                            sendType = 0x01;
                            break;
                        case "前进":
                            sendType = 0x12;
                            break;
                        case "后退":
                            sendType = 0x22;
                            break;
                        case "左移":
                            sendType = 0x14;
                            break;
                        case "右移":
                            sendType = 0x24;
                            break;
                        case "静止":
                            sendType = 0x08;
                            break;
                    }
                }
            }
            //获取画面PID三项数据
            float P = float.Parse(this.txtP.Text);
            float I = float.Parse(this.txtI.Text);
            float D = float.Parse(this.txtD.Text);

            //下发参数
            Pid(sendType, P, I, D, 0x00,"PID");
        }

        /// <summary>
        /// 写入FLASH
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void btnWrite_Click(object sender, EventArgs e)
        {
            //获取画面PID选项
            byte sendType = 0x00;
            //获取画面PID三项数据
            float P = 0F;
            float I = 0F;
            float D = 0F;
            //写入FLASH
            byte sendFlash = 0xFF;

            //下发参数
            Pid(sendType, P, I, D, sendFlash, "FLASH");
        }
    }
}
