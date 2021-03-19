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
