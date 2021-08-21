from math import pi, atan
import time

import rov
import visual_info


class Auv(rov.Rov):
    """BUAA AUV robot for seafood auto collecting

    should be created with:
      >>> with Auv() as auv:
      >>>    pass

    Attributes
    ----------
    state : str
        current automation workflow state of AUV
    """

    def __init__(self, state: str = 'initial'):
        self.state = state
        self.target = visual_info.Target()
        self.arm = visual_info.Arm()
        self.grasp_state = 'idle'
        self.cruise_periods = 2
        self.cruise_time = 0.0
        # the period is 7s, keys are the last time of that move [key_n-1, key_n)
        self.cruise_path = {
            1: [0, 0, -0.5, -0.99],
            2: [0.7, 0, 0, 0.3],
            3: [0, 0, 0.5, -0.99],
            4: [0.7, 0, 0, 0.3],
            5: [0, 0, 0.5, -0.99],
            6: [0.7, 0, 0, 0.3],
            7: [0, 0, -0.5, -0.99],
        }
        self.aim_chances = [4] * 2  # FIXME: may adjust
        super().__init__()

    def __enter__(self):
        return super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        return super().__exit__(exc_type, exc_value, traceback)

    def start(self):
        return super().start()

    def kill(self):
        return super().kill()

    def int2u8list(self, value: int, n: int):
        return super().int2u8list(value, n)

    def write(self, id: int, data: list):
        return super().write(id, data)

    def read(self, id: int, request_needed: bool) -> list:
        return super().read(id, request_needed)

    def set_led(self, value: float):
        return super().set_led(value)

    def set_Vx(self, value: float):
        return super().set_Vx(value)

    def set_Vy(self, value: float):
        return super().set_Vy(value)

    def set_Vz(self, value: float):
        return super().set_Vz(value)

    def set_direction(self, value: float):
        return super().set_direction(value)

    def set_move(self, velocity: list):
        return super().set_move(velocity)

    def get_sensors_data(self):
        return super().get_sensors_data()

    def land(self) -> str:
        """坐底
        """
        self.set_Vz(-1)
        if self.depth_sensor.is_landed:
            print('[AUV] landed, start grasping')
            return 'grasp'
        else:
            return 'land'

    def grasp(self) -> str:
        """抓取
        """
        # TODO: change to judge by Manipulator.inverse_kinematics_simplified
        if self.target.roi_check():
            if not self.arm.arm_is_working:
                self.grasp_state = 'ready'
                return 'grasp'
            else:
                if self.arm.chances[0]:
                    if time.time() - self.arm.start_time > self.arm.time_limit:
                        self.arm.chances[0] -= 1
                    self.grasp_state = 'activated'
                    return 'grasp'
        # has no more target in thresh range / no more chances
        self.grasp_state = 'idle'
        self.arm.chances[0] = self.arm.chances[1]  # reset chances
        # 往前荡一下, 确保目标进袋
        self.set_Vx(1)
        time.sleep(1)
        # float up to cruise hight
        self.set_Vz(1)
        time.sleep(0.5)
        print('[AUV] grasp done, start cruise')
        return 'cruise'

    def cruise(self) -> str:
        """巡航
        """
        if self.target.has_target:
            self.cruise_time = 0
            # 发现目标后一个后撤步!
            self.set_move([-0.7, 0, 0, -0.99])
            time.sleep(1)  # 1s
            print('[AUV] target found, start aiming')
            return 'aim'
        # if reach time limit
        elif time.time() - self.cruise_time > list(self.cruise_path.keys())[-1] * self.cruise_periods:
            self.cruise_time = 0
            print('[AUV] time out, start landing')
            return 'land'
        else:
            key = list(self.cruise_path.keys())[0]
            if self.cruise_time == 0:
                self.cruise_time = time.time()
            else:
                time_count = time.time() - self.cruise_time
                for t in self.cruise_path:
                    if time_count < t:
                        key = t
                        break
            self.set_move(self.cruise_path[key])
            return 'cruise'

    def aim(self) -> str:
        """瞄准, 移动至目标处
        """
        grasp_thresh_x = self.target.roi_thresh[0]
        grasp_thresh_y = self.target.roi_thresh[1]
        Vy = 0
        omega = 0
        if not self.target.has_target:
            # 目标丢失, 转坐底
            self.aim_chances[0] = self.aim_chances[1]
            print('[AUV] target lost, start landing')
            return 'land'
        else:
            # 最多4次调整机会
            if self.aim_chances[0]:
                offset_y = self.target.roi_offset[1]
                dx = 0.5 - self.target.center[0]
                dy = offset_y - self.target.center[1]  # 线之上为正
                # 以底部中间处为原点, y多减0.01保证分母不为零
                theta = atan((self.target.center[0] - 0.5) / (self.target.center[1] - 1 - 0.01))
                omega = theta / (pi / 2) * 0.99
                if not self.depth_sensor.is_landed:
                    # 级联阈值，粗调+细调，保证目标一直在视野范围内
                    # FIXME: a sleep may needed to limit the fps (or not)
                    if abs(dx) > 0.35:
                        omega = omega * 1.3  # 限制大小
                    elif 0.4 <= dy < offset_y:
                        Vy = dy * 1.1
                    elif 0.3 <= dy < 0.4:  # 离得近一些速度放小
                        Vy = dy * 0.9
                    elif abs(dx) > 0.25:
                        omega = omega * 0.8
                    else:  # 第二阈值框
                        if abs(dy) > grasp_thresh_y:
                            Vy = dy * 0.8
                        elif abs(dx) > grasp_thresh_x:
                            omega = omega * 1.2
                    # print(f'[AUV] {}')  # FIXME: what is this
                    self.set_move([0, max(min(Vy, 1), -1), 0, max(min(omega, 1), -1)])
                    return 'aim'
                else:  # 坐底后考虑是否起跳调整位置
                    if self.target.roi_check():  # 检查位置阈值
                        self.aim_chances[0] = self.aim_chances[1]
                        print('[AUV] ready for grasping, start grasping!')
                        return 'grasp'
                    else:
                        if abs(dx) > 0.35:
                            omega = omega * 1
                        elif abs(dy) > grasp_thresh_y:
                            if abs(dy) > 0.3:
                                Vy = dy * 1.8
                            else:
                                Vy = dy * 1.4
                        elif abs(dx) > grasp_thresh_x:
                            omega = omega * 0.8
                        # 上浮
                        # print(f'[AUV] {}')  # FIXME: what is this
                        self.aim_chances[0] -= 1
                        self.set_move([0, max(min(Vy, 1), -1), 0, max(min(omega, 1), -1)])
                        time.sleep(0.5)  # 上浮0.5s
                        return 'aim'
            else:
                # 放弃瞄准, 转坐底, 然后会转抓取, 如果还是没有东西会转巡航
                self.aim_chances[0] = self.aim_chances[1]
                print('[AUV] give up aiming, start landing')
                return 'land'

    def state_machine(self) -> str:
        cases = {
            'aim': self.aim,
            'cruise': self.cruise,
            'grasp': self.grasp,
            'land': self.land,
        }
        self.state = cases[self.state]()
        return self.grasp_state


if __name__ == '__main__':
    import socket
    import yaml
    from asciimatics.screen import Screen
    import controller

    VISUAL_SERVER_PORT = 8080

    def screen_main(screen: Screen):
        """the function to call once the screen has been created
        """
        quit_flag = False
        switch = False
        with Auv() as auv:
            ctrl = controller.Controller(auv, screen)
            # AUV主循环
            while True:
                # 根据键盘事件控制AUV, 并刷新屏幕内容
                quit_flag = ctrl.key_check()
                ctrl.printScreen()
                # 更新target, arm数据
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as visual_socket:
                    try:
                        visual_socket.connect(('127.0.0.1', VISUAL_SERVER_PORT))
                    except ConnectionRefusedError:
                        print('[Visual Info Client] lost connection')
                        # TODO: 启动/重启visual_info server
                        continue
                    # send flags to visual server
                    # threads_quit_flag: 2; arm_is_working: 1
                    visual_socket.send(bytes(str(quit_flag * 2 + auv.arm.arm_is_working).encode()))
                    # receive data from ROV then update target and arm
                    visual_info_dict = yaml.load(visual_socket.recv(1024), Loader=yaml.Loader)
                # quit_flag置1后待运行到下一循环, 将quit_flag发送给visual_server后再break
                if quit_flag:
                    if switch:
                        break
                    else:
                        switch = True
                # update AUV data
                auv.target.update(visual_info_dict["target"])
                auv.arm.update(visual_info_dict["arm"])
                auv.get_sensors_data()  # 主要是获取深度
                # 状态机状态跳转并给出抓取判断
                grasp_state = auv.state_machine()
                # 软体臂控制
                if grasp_state == 'ready':
                    # FIXME: 软体臂可以开始了. 这个分支好像可以删掉了
                    auv.arm.arm_is_working = True
                    auv.arm.start_time = time.time()
                elif grasp_state == 'activated':
                    # TODO: 软体臂发一次指令
                    pass
                elif grasp_state == 'idle':
                    auv.arm.arm_is_working = False

    Screen.wrapper(screen_main)
