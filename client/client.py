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
    import time
    t = time.time()
    quit_flag = False
    switch = False
    try:
        uart = serial.Serial('/dev/ttyUSB0', baudrate=9600)
        print('[Uart] connected')
    except (FileNotFoundError, serial.SerialException):
        uart = None
    with rov.Rov() as rov:
        # ROV主循环
        while True:
            # TODO: interface needed, for at least quit
            if time.time() - t > 30:
                quit_flag = True
            else:
                quit_flag = False
            # 更新target, arm数据
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.connect(('127.0.0.1', VISUAL_SERVER_PORT))
                except ConnectionRefusedError:
                    print('[Visual Info Client] lost connection')
                    continue
                # send flags to ROV
                # threads_quit_flag: 2; arm_is_working: 1
                s.send(bytes(str(quit_flag * 2 + rov.arm.arm_is_working).encode()))
                # receive data from ROV then update target and arm
                visual_info_dict = yaml.load(s.recv(1024), Loader=yaml.Loader)
            if quit_flag:
                if switch:
                    break
                else:
                    switch = True
            # update ROV data
            rov.target.update(visual_info_dict["target"])
            rov.arm.update(visual_info_dict["arm"])
            rov.get()  # 主要是获得深度
            rov.depth_sensor.land_check()
            # switch case
            grasp_state = rov.state_machine()
            if grasp_state == 'ready':
                if uart is not None:
                    uart.write('!!')
                    rov.arm.arm_is_working = True
                    rov.arm.start_time = time.time()
            elif grasp_state == 'activated':
                if uart is not None:
                    # FIXME: check the message format
                    uart.write(f'#{str((rov.target.center[0] - rov.arm.marker_position[0]) * 100)},\
                    {str((rov.target.center[1] - rov.arm.marker_position[1]) * 100)}')
            elif grasp_state == 'idle':
                rov.arm.arm_is_working = False
    if uart is not None:
        uart.close()
