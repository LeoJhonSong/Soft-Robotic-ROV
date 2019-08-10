/* ***********************************************
 * Author		:  kingthy
 * Email		:  kingthy@gmail.com
 * DateTime		:  2009-3-27 19:59:15
 * Description	:  游戏手柄的事件参数
 *
 * ***********************************************/

using System;
using System.Collections.Generic;
using System.Text;

namespace SoftwareArmROV
{
    /// <summary>
    /// 游戏手柄的事件参数
    /// </summary>
    public class JoystickEventArgs : EventArgs
    {
        /// <summary>
        /// 游戏手柄的事件参数
        /// </summary>
        /// <param name="joystickId">手柄Id</param>
        /// <param name="buttons">按钮</param>
        public JoystickEventArgs(int joystickId, JoystickButtons buttons)
        {
            this.JoystickId = joystickId;
            this.Buttons = buttons;
        }
        /// <summary>
        /// 手柄Id
        /// </summary>
        public int JoystickId { get; private set; }
        /// <summary>
        /// 按钮
        /// </summary>
        public JoystickButtons Buttons { get; private set; }
    }

    /// <summary>
    /// 游戏手柄的按钮定义
    /// </summary>
    [Flags]
    public enum JoystickButtons
    {
        //没有任何按钮
        None = 0x0,
        UP = 0x01,
        Down = 0x02,
        Left = 0x04,
        Right = 0x08,
        B1 = 0x10,
        B2 = 0x20,
        B3 = 0x40,
        B4 = 0x80,
        B5 = 0x100,
        B6 = 0x200,
        B7 = 0x400,
        B8 = 0x800,
        B9 = 0x1000,
        B10 = 0x2000
    }
}
