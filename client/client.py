#! /usr/bin/env python3

'''Visual info client module
when run as main, start visual info client, ROV server
'''

import socket
import yaml
import rovClient

VISUAL_SERVER_PORT = 8080


class Target(object):
    def __init__(self):
        self.has_target = False
        self.target_class = 0
        self.id = -1
        self.center = [0, 0]
        self.shape = [0, 0]

    def update(self, target_dict):
        self.has_target = bool(target_dict["has_target"])
        self.target_class = target_dict["target_class"]
        self.id = target_dict["id"]
        self.center = [target_dict["center"]["x"], target_dict["center"]["y"]]
        self.shape = [target_dict["shape"]["width"], target_dict["shape"]["height"]]


class Arm(object):
    def __init__(self):
        self.arm_is_working = True
        self.has_marker = False
        self.marker_position = [0, 0]

    def update(self, arm_dict):
        self.has_marker = arm_dict["has_marker"]
        self.marker_position = [arm_dict["position"]["x"], arm_dict["position"]["y"]]


if __name__ == '__main__':
    import time
    t = time.time()
    target = Target()
    arm = Arm()
    quit_flag = False
    switch = False
    with rovClient.Rov() as rov:
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
            # TODO: update rov status
            pass
            # switch case
            rov.switch()
