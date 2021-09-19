class Target():
    """info of target detected by visual_info server
    """
    def __init__(self):
        self.has_target = False
        self.target_class = 0
        self.id = -1
        self.center = (0.0, 0.0)
        self.shape = [0, 0]
        self.roi_offset = [0.5, 0.7]  # x, y
        self.roi_thresh = [0.2, 0.15]  # x, y

    def update(self, target_dict):
        self.has_target = bool(target_dict["has_target"])
        self.target_class = target_dict["target_class"]
        self.id = target_dict["id"]
        self.center = (target_dict["center"]["x"], target_dict["center"]["y"])
        self.shape = [target_dict["shape"]["width"], target_dict["shape"]["height"]]

    def roi_check(self) -> bool:
        """check if target in the ROI range

        Returns
        -------
        bool
            if target in range, return True, else False
        """
        x_check = abs(self.center[0] - self.roi_offset[0]) < self.roi_thresh[0]
        y_check = abs(self.center[1] - self.roi_offset[1]) < self.roi_thresh[1]
        if x_check and y_check:
            return True
        else:
            return False


class Arm():
    """info of arm detected by visual_info server
    """
    def __init__(self):
        self.arm_is_working = True
        self.has_marker = False
        self.marker_position = (0.0, 0.0)
        self.start_time = 0.0
        self.time_limit = 70  # 70s
        self.chances = [2] * 2  # use the the second to store the total chances

    def update(self, arm_dict):
        self.has_marker = arm_dict["has_marker"]
        self.marker_position = (arm_dict["position"]["x"], arm_dict["position"]["y"])
