import os
import time
from typing import Tuple

import can

from . import ms5837
from .utils import tprint


class Gyro(object):
    """Gyroscope sensor class

    Attributes
    ----------
    x : float
        roll (横滚角)
    y : float
        pitch (俯仰角)
    z : float
        yaw (偏航角)
    """

    def __init__(self) -> None:
        self.x = 0
        self.y = 0
        self.z = 0

    def update(self, gyro_list: list) -> None:
        self.x = gyro_list[0]
        self.y = gyro_list[1]
        self.z = gyro_list[2]


class Depth_sensor(object):
    """depth sensor for ROV

    Attributes
    ----------
    depth : float
        in meters
    relative_pressure: float
        in kPa, relative to initial pressure
    """

    def __init__(self):
        self.depth = 0.0
        self.old_depth = 0.0
        self.relative_pressure = 0  # unit: kPa
        self.time = 0
        self.time_thresh = 2  # unit: s
        self.diff_thresh = 0.03  # 3cm
        self.is_landed = False
        self.sensor = ms5837.MS5837_30BA()
        self.sensor.init()
        self.sensor.setFluidDensity(ms5837.DENSITY_SALTWATER)
        self.reset()
        tprint(f'📏 depth sensor initialed')

    def reset(self):
        """reset initial depth and pressure of depth sensor
        """
        self.sensor.read()
        self.init_depth = self.sensor.depth()  # unit: m
        self.init_pressure = self.sensor.pressure(ms5837.UNITS_kPa)  # unit: kPa

    def update(self) -> None:
        self.sensor.read()
        depth = max(0, self.sensor.depth() - self.init_depth)
        self.relative_pressure = max(0, self.sensor.pressure(ms5837.UNITS_kPa) - self.init_pressure)
        # check if landed onto the sea bed
        if abs(self.old_depth - depth) < self.diff_thresh:
            if self.time == 0:
                self.time = time.time()
            if time.time() - self.time > self.time_thresh:
                if not self.is_landed:
                    tprint('📏 landed / hovering!')
                    self.is_landed = True
        else:  # 当深度变化幅度超过阈值, 判定未坐底并归零稳定计时
            self.old_depth = depth
            if self.is_landed:
                tprint('📏 leaving seabed / leaving hovering state')
            self.time = 0
            self.is_landed = False
        self.old_depth = self.depth
        self.depth = depth


# set can config for Jetson
can_config = {
    'bustype': 'socketcan',
    'channel': 'can0',
    'bitrate': 1000000  # 1000k bits/s
}
# 设置远程帧及返回的数据帧的仲裁id
depth = 0x42  # 深度
pitch = 0x18  # Pitch角
roll = 0x19  # Roll角
yaw = 0x1a  # Yaw角
# 设置数据帧的仲裁id
speed = 0x11  # 设置各方向速度 (设置电机速度)
led1 = 0x21  # 设置传感器1板照明PWM
led2 = 0x31  # 设置传感器2板照明PWM
depthPID = 0x51  # 设置深度PID
fbPID = 0x52  # 设置前后PID (forward/backward)
swPID = 0x53  # 设置侧移PID (swing)
flashSig = 0x54  # 设置写flash信号
hovPID = 0x55  # 设置悬停PID (hover)
# set param of sensors
led_pwm_max = 950


class Rov(object):
    """BUAA ROV (Remotely Operated Vehicle) driving class running in a Jetson
    AGX Xavier (as VCU) on CAN bus
    """

    def __init__(self):
        # set motors speed controlled by closed-loop (PID): 0xaa or open-loop: 0x55
        self.is_closed_loop = 0x55
        # UI测试模式/手柄遥控模式: 0/1
        self.control_mode = 0xaa  # 手柄遥控模式为0xaa, UI测试模式为0x55
        # initial sensors
        self.depth_sensor = Depth_sensor()
        self.gyro = Gyro()
        # turn lights on
        self.start()
        for i in range(2):
            time.sleep(0.1)
            self.set_led(1)
            time.sleep(0.1)
            self.set_led(0)
        tprint('🚀 ROV started')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.reset()
        self.kill()

    def start(self):
        # set Jetson AGX Xavier's corresponding register values to configure CAN
        # I/O pins as CAN functionalities. see:
        # https://forums.developer.nvidia.com/t/enable-can-on-xavier/65869/2
        os.system('''
        # for can0
        sudo busybox devmem 0xc303000 32 0x0000c400
        sudo busybox devmem 0xc303008 32 0x0000c458
        # for can1
        sudo busybox devmem 0xc303010 32 0x0000c400
        sudo busybox devmem 0xc303018 32 0x0000c458
        # insert required kernel modules
        sudo modprobe can
        sudo modprobe can-raw
        sudo modprobe mttcan
        ''')
        # create socketcan in system
        os.system('sudo ip link set ' + can_config['channel'] + ' type can bitrate ' + str(can_config['bitrate']))
        os.system('sudo ifconfig ' + can_config['channel'] + ' up')
        # initial can bus instance
        self.bus = can.interface.Bus(bustype=can_config['bustype'], channel=can_config['channel'], bitrate=can_config['bitrate'])

    def kill(self):
        self.bus.shutdown()
        # turn down socketcan
        os.system('sudo ifconfig ' + can_config['channel'] + ' down')

    def int2u8list(self, value: int, n: int):
        """convert int to a list of int in range of 0~255

        Parameters
        ----------
        value : int
        n : int
            the length of the list should be
        """
        return list((value).to_bytes(n, 'big'))

    def write(self, id: int, data: list):
        msg = can.Message(arbitration_id=id, data=data, is_extended_id=False)
        self.bus.send(msg)

    def read(self, id: int, request_needed: bool) -> list:
        if request_needed:
            remote_msg = can.Message(arbitration_id=id, is_extended_id=False, is_remote_frame=True)
            self.bus.send(remote_msg)
        # continuously read message from bus
        for msg in self.bus:
            if msg.arbitration_id == id and msg.is_remote_frame is False:
                # if remote frame sent, this msg is the data frame we requested from source node
                return msg.data
        return []  # actually will not be reached, just not to break the type hint

    def set_led(self, value: float):
        """set brightness of all four LEDs

        Parameters
        ----------
        value : float
            in range [0, 1]
        """
        self.write(led2, self.int2u8list(int(value * led_pwm_max), 2))
        tprint(f'🚀 ROV led brightness set to {value * 100:.02f}%')

    def set_Vx(self, value: float):
        """set Vx and clear others

        Parameters
        ----------
        value : float
            in range (backward)[-1, 1](forward)
        """
        # 先置零电机转速, 防止电机立即反转损伤电机
        self.write(speed, [127, 127, 127, 127, self.control_mode, self.is_closed_loop])
        time.sleep(0.01)
        self.write(speed, [int(127 + 127 * value), 127, 127, 127, self.control_mode, self.is_closed_loop])
        if value:
            tprint(f'🚀 ROV going {"forward" if value > 0 else "backward"} with {int(abs(value) * 100):.02f}% speed')
        else:
            tprint('🚀 ROV stopped')

    def set_Vy(self, value: float):
        """set Vy and clear others

        Parameters
        ----------
        value : float
            in range (right)[-1, 1](left)
        """
        # 先置零电机转速, 防止电机立即反转损伤电机
        self.write(speed, [127, 127, 127, 127, self.control_mode, self.is_closed_loop])
        time.sleep(0.01)
        self.write(speed, [127, int(127 + 127 * value), 127, 127, self.control_mode, self.is_closed_loop])
        if value:
            tprint(f'🚀 ROV going {"left" if value > 0 else "right"} with {int(abs(value) * 100):.02f}% speed')
        else:
            tprint('🚀 ROV stopped')

    def set_Vz(self, value: float):
        """set Vz and clear others

        Parameters
        ----------
        value : float
            in range (down)[-1, 1](up)
        """
        # 先置零电机转速, 防止电机立即反转损伤电机
        self.write(speed, [127, 127, 127, 127, self.control_mode, self.is_closed_loop])
        time.sleep(0.01)
        self.write(speed, [127, 127, 127, int(127 + 127 * value), self.control_mode, self.is_closed_loop])
        if value:
            tprint(f'🚀 ROV going {"up" if value > 0 else "down"} with {int(abs(value) * 100):.02f}% speed')
        else:
            tprint('🚀 ROV stopped')

    def set_steer(self, value: float):
        """set steer and clear others

        Parameters
        ----------
        value : float
            in range (left)[-1, 1](right)
        """
        # 先置零电机转速, 防止电机立即反转损伤电机
        self.write(speed, [127, 127, 127, 127, self.control_mode, self.is_closed_loop])
        time.sleep(0.01)
        self.write(speed, [127, 127, int(127 + 127 * value), 127, self.control_mode, self.is_closed_loop])
        if value:
            tprint(f'🚀 ROV turning {"right" if value > 0 else "left"} with {int(abs(value) * 100):.02f}% speed')
        else:
            tprint('🚀 ROV stopped')

    def set_move(self, velocity: Tuple[float, float, float, float]):
        """set multi-direction values

        Parameters
        ----------
        velocity : tuple
            a tuple of (Vx, Vy, steer, Vz) in range [-1, 1]. It seems the ROV
            could do all four directions move at the same time
        """
        # 先置零电机转速, 防止电机立即反转损伤电机
        self.write(speed, [127, 127, 127, 127, self.control_mode, self.is_closed_loop])
        time.sleep(0.01)
        self.write(speed, [int(127 + 127 * v) for v in velocity] + [self.control_mode, self.is_closed_loop])
        tprint(f'🚀 ROV Vx: {100 * velocity[0]:.02f}%, Vy: {100 * velocity[1]:.02f}%, Vz: {100 * velocity[-1]:.02f}%, steer: {100 * velocity[2]:.02f}%')

    def reset(self):
        """reset ROV led, motors
        """
        self.set_move((0, 0, 0, 0))
        tprint('🚀 ROV motors reset 👌')

    def get_sensors_data(self):
        """get the latest data from sensors
        """
        self.depth_sensor.update()
        # 这个角度读取有一定可能卡住
        # gyro_list = [self.read(d, True) for d in [roll, pitch, yaw]]  # roll, pitch, yaw
        # gyro_list = [f"{int.from_bytes(i, 'big'):06x}" for i in gyro_list]  # convert data into 'SXXXYY' format strings
        # gyro_list = [(-1 if int(i[0]) else 1) * (int(i[1:4]) + int(i[4:]) / 100) for i in gyro_list]  # calculated to float
        # self.gyro.update(gyro_list)


if __name__ == '__main__':
    # for simple test
    with Rov() as robot:
        while True:
            command, value = input('Vx [-1, 1], Vy [-1, 1], Vz [-1, 1], led [0, 1], steer [-1, 1]').split(',')
            eval(f'robot.set_{command}({value})')
            # robot.get_sensors_data()
            # print(robot.depth_sensor.depth)
