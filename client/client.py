#! /usr/bin/env python3

'''Visual info client module
when run as main, start visual info client, ROV server
'''

import socket
import time

import serial
import yaml

import rov
import visual_info

VISUAL_SERVER_PORT = 8080


if __name__ == '__main__':
    t = time.time()
    quit_flag = False
    switch = False
    try:
        uart = serial.Serial('/dev/ttyUSB0', baudrate=9600)
        print('[Uart] connected')
    except (FileNotFoundError, serial.SerialException):
        uart = None
    with rov.Rov() as robot:
        # ROV主循环
        while True:
            # TODO: interface needed, for at least quit
            if time.time() - t > 30:
                quit_flag = True
            else:
                quit_flag = False
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
            # quit in next loop when quit_flag shows up
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
            # switch case
            grasp_state = robot.state_machine()
            if grasp_state == 'ready':
                if uart is not None:
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
