#! /usr/bin/env python3

'''Visual info client module
when run as main, start visual info client, ROV server
'''

import socket

import serial
import yaml

import rov
import visual_info

VISUAL_SERVER_PORT = 8080


if __name__ == '__main__':
    import time
    t = time.time()
    target = visual_info.Target()
    arm = visual_info.Arm()
    quit_flag = False
    switch = False
    try:
        uart = serial.Serial('/dev/ttyUSB0', baudrate=9600)
        print('[Uart] connected')
    except FileNotFoundError:
        uart = None
    with rov.Rov() as rov:
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
                s.send(bytes(str(quit_flag * 2 + arm.arm_is_working).encode()))
                # receive data from ROV then update target and arm
                visual_info_dict = yaml.load(s.recv(1024), Loader=yaml.Loader)
                target.update(visual_info_dict["target"])
                arm.update(visual_info_dict["arm"])
            if quit_flag:
                if switch:
                    break
                else:
                    switch = True
            # update ROV data
            rov.target = target
            rov.arm = arm
            # switch case
            grasp_state = rov.state_machine()
            if grasp_state == 'ready':
                if uart is not None:
                    uart.write('!!')
                    arm.arm_is_working = True
            elif grasp_state == 'started':
                if uart is not None:
                    # FIXME: check the message format
                    uart.write(f'#{str((target.center[0] - arm.marker_position[0]) * 100)},\
                    {str((target.center[1] - arm.marker_position[1]) * 100)}')
            elif grasp_state == 'idle':
                arm.arm_is_working = False
    if uart is not None:
        uart.close()
