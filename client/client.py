#! /usr/bin/env python3

'''
Visual info client
'''

import socket
import yaml

SERVER_PORT = 8080


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
    while True:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.connect(('127.0.0.1', SERVER_PORT))
        except ConnectionRefusedError:
            print('[Client] lost connection')
        else:
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
