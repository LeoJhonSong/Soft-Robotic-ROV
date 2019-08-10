using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace SoftwareArmROV
{
   public class JosystickSend
    {
       //前后
        public int Forward_Back_Value_R { get; set; } 
       //侧移
        public int Side_Shift_Value_X { get; set; }
       //旋转
        public int Direc_Value_Z { get; set; }
       //上下
        public int UpDown_Value_Y { get; set; }
    }
}
