#! /usr/bin/env python3

'''Visual info client module
when run as main, start visual info client, ROV server
'''

import socket
import yaml
import visual_info
import rov

VISUAL_SERVER_PORT = 8080


if __name__ == '__main__':
    import time
    t = time.time()
    target = visual_info.Target()
    arm = visual_info.Arm()
    quit_flag = False
    switch = False
    with rov.Rov() as rov:
        while True:
            # 更新target, arm数据
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                s.connect(('127.0.0.1', VISUAL_SERVER_PORT))
            except ConnectionRefusedError:
                print('[Client] lost connection')
                continue
            # threads_quit_flag: 2; arm_is_working: 1
            if time.time() - t > 30:
                quit_flag = True
            else:
                quit_flag = False
            s.send(bytes(str(quit_flag * 2 + arm.arm_is_working).encode()))
            visual_info_dict = yaml.load(s.recv(1024), Loader=yaml.Loader)
            target.update(visual_info_dict["target"])
            arm.update(visual_info_dict["arm"])
            s.close()
            if quit_flag:
                if switch:
                    break
                else:
                    switch = True
            # update ROV data
            rov.target = target
            rov.arm = arm
            # switch case
            rov.switch()
