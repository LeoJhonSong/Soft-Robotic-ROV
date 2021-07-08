#! /usr/bin/env python3

'''Visual info client module
when run as main, start visual info client, ROV server
'''

import socket
import time

import serial
import yaml

from asciimatics.screen import Screen

import rov
import controller

VISUAL_SERVER_PORT = 8080


def screen_main(screen: Screen):
    """the function to call once the screen has been created
    """
    quit_flag = False
    switch = False
    try:
        uart = serial.Serial('/dev/ttyUSB0', baudrate=9600)
        print('[Uart] connected')
    except (FileNotFoundError, serial.SerialException):
        uart = None
    with rov.Rov() as robot:
        ctrl = controller.Controller(robot, screen)
        # ROV主循环
        while True:
            # 根据键盘事件控制ROV, 并刷新屏幕内容
            quit_flag = ctrl.key_check()
            ctrl.printScreen()
            # 更新target, arm数据
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as visual_socket:
                try:
                    visual_socket.connect(('127.0.0.1', VISUAL_SERVER_PORT))
                except ConnectionRefusedError:
                    print('[Visual Info Client] lost connection')
                    continue
                # send flags to visual server
                # threads_quit_flag: 2; arm_is_working: 1
                visual_socket.send(bytes(str(quit_flag * 2 + robot.arm.arm_is_working).encode()))
                # receive data from ROV then update target and arm
                visual_info_dict = yaml.load(visual_socket.recv(1024), Loader=yaml.Loader)
            # quit_flag置1后待运行到下一循环, 将quit_flag发送给visual_server后再break
            if quit_flag:
                if switch:
                    break
                else:
                    switch = True
            # update ROV data
            robot.target.update(visual_info_dict["target"])
            robot.arm.update(visual_info_dict["arm"])
            robot.get()  # 主要是获得深度
            robot.depth_sensor.land_check()
            # 状态机状态跳转并给出抓取判断
            grasp_state = robot.state_machine()
            # 软体臂控制
            if grasp_state == 'ready':
                if uart is not None:
                    # TODO: 接入软体臂算法
                    uart.write('!!')
                    robot.arm.arm_is_working = True
                    robot.arm.start_time = time.time()
            elif grasp_state == 'activated':
                if uart is not None:
                    # FIXME: check the message format
                    uart.write(f'#{str((robot.target.center[0] - robot.arm.marker_position[0]) * 100)},\
                    {str((robot.target.center[1] - robot.arm.marker_position[1]) * 100)}')
            elif grasp_state == 'idle':
                robot.arm.arm_is_working = False
    if uart is not None:
        uart.close()


if __name__ == '__main__':
    Screen.wrapper(screen_main)
