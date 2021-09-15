from math import pi, atan
import time

from . import rov
from .import visual_info
from . import manipulate


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
        self.visual_arm = visual_info.Arm()
        self.grasp_state = 'idle'
        self.cruise_time = 0.0
        # the period is 7s, keys are the last time of that move [key_n-1, key_n)
        self.cruise_path = {
            1: (0, 0, -0.7, -0.99),
            2: (0.7, 0, 0, 0.1),
            3: (0, 0, 0.5, -0.99),
            4: (0.7, 0, 0, 0.1),
            5: (0, 0, -0.5, -0.99),
            6: (0.7, 0, 0, 0.1),
            7: (0, 0, 0.5, -0.99),
            8: (0.7, 0, 0, 0.1),
            9: (0, 0, -0.7, -0.99),
            10: (0.7, 0, 0, 0.1),
            11: (0, 0, 0.5, -0.99),
            12: (0.7, 0, 0, 0.1),
            13: (0, 0, -0.7, -0.99),
        }
        self.aim_chances = [4, 4]  # FIXME: 需要调参: 尝试次数
        self.arm = manipulate.Manipulator()
        super().__init__()

    def reset(self):
        self.arm.release()
        self.state = 'initial'
        self.grasp_state = 'idle'
        return super().reset()

    def get_sensors_data(self):
        super().get_sensors_data()
        self.arm.water_pressure = self.depth_sensor.relative_pressure  # kPa

    def initial(self):
        return 'initial'

    def land(self) -> str:
        """坐底
        """
        self.set_Vz(-1)
        if self.depth_sensor.is_landed:
            print('🤿 AUV landed, switch to grasp state')
            return 'grasp'
        else:
            return 'land'

    def grasp(self) -> str:
        """抓取
        """
        time.sleep(1)  # 等待画面稳定
        # no more chances
        if self.visual_arm.chances[0] == 0:
            self.grasp_state = 'idle'
            self.visual_arm.chances[0] = self.visual_arm.chances[1]  # reset chances
            print('🤿 🤷 AUV give up grasping, start cruise')
            return 'cruise'
        # 当还有机会且阈值框内有目标
        if self.target.roi_check():
            if not self.visual_arm.arm_is_working:
                self.grasp_state = 'ready'
                print('🤿 🎯 target in range, start grasping')
            return 'grasp'
        else:
            if self.grasp_state in ['ready', 'activated']:  # 正在抓取, 且还有机会. 此时目标很可能被手臂挡住, 不再关注实时目标
                if time.time() - self.visual_arm.start_time > self.visual_arm.time_limit:
                    self.visual_arm.chances[0] -= 1
                    # 收回手臂后再次识别并抓取
                    self.grasp_state = 'ready'
                    self.reset()
                    time.sleep(2)
                else:
                    print(f'💪 🕐 time cost: {time.time() - self.visual_arm.start_time}')
                    self.grasp_state = 'activated'
                return 'grasp'
            elif self.target.has_target:  # 有目标但目标在阈值框外, 而且没有在抓
                print(f'🤿 👀 target not in range: {self.target.center}, try to aim')
                self.grasp_state = 'idle'
                self.visual_arm.chances[0] = self.visual_arm.chances[1]  # reset chances
                return 'aim'
            else:
                # has no more target in thresh range
                self.grasp_state = 'idle'
                # 进行了抓取尝试又没有目标了说明抓到了
                if self.visual_arm.chances[0] != self.visual_arm.chances[1]:
                    print('🎉 target collected')
                    # 往前荡一下, 确保目标进袋
                    self.set_move((1, 0, 0, 1))
                    self.arm.reset()
                    time.sleep(1)
                self.visual_arm.chances[0] = self.visual_arm.chances[1]  # reset chances
                print('🤿 AUV grasp done, start cruise')
                return 'cruise'

    def cruise(self) -> str:
        """巡航
        """
        if self.target.has_target:
            self.cruise_time = 0
            # 发现目标后一个后撤步!
            self.set_move((-0.7, 0, 0, 0))
            time.sleep(0.2)  # 0.2s
            self.set_Vx(0)
            print('🤿 👀 AUV target found, start aiming')
            return 'aim'
        # if reach time limit
        elif self.cruise_time != 0 and time.time() - self.cruise_time > list(self.cruise_path.keys())[-1]:
            print(f'🤿 ⏰ AUV time out, start landing {time.time() - self.cruise_time}')
            self.cruise_time = 0
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
        xy坐标系为图像坐标系
        """
        grasp_thresh_x = self.target.roi_thresh[0]
        grasp_thresh_y = self.target.roi_thresh[1]
        Vy = 0
        omega = 0
        if not self.target.has_target:
            # 目标丢失, 转坐底
            self.aim_chances[0] = self.aim_chances[1]
            print('🤿 AUV target lost, start landing')
            return 'land'
        else:
            # 最多4次调整机会
            if self.aim_chances[0]:
                offset_x = self.target.roi_offset[0]
                offset_y = self.target.roi_offset[1]
                dx = offset_x - self.target.center[0]
                dy = offset_y - self.target.center[1]
                # 以底部中间处为原点, y多减0.01保证分母不为零
                theta = atan((self.target.center[0] - offset_x) / (1 - self.target.center[1] + 0.001))
                omega = theta / (pi / 2) * 0.99
                if not self.depth_sensor.is_landed:
                    # 级联阈值，粗调+细调，优先旋转, 保证目标一直在视野范围内
                    if abs(dx) > 1.5 * grasp_thresh_x:  # x一级阈值框
                        omega = omega * 0.8  # 限制大小
                        Vy = 0
                    elif abs(dy) > 1.5 * grasp_thresh_y:  # y一级阈值框
                        Vy = dy * 2.1
                        omega = 0
                    elif grasp_thresh_y <= abs(dy) <= 1.5 * grasp_thresh_y:  # y二级阈值框, 离得近一些速度放小
                        Vy = dy * 1.8
                        omega = 0
                    elif grasp_thresh_x <= abs(dx) <= 1.5 * grasp_thresh_x:  # x二级阈值框
                        omega = omega * 0.6
                        Vy = 0
                    print(f'🤿 AUV aiming! try: {self.aim_chances[-1] - self.aim_chances[0]}')
                    print(f'❌ target: {self.target.center}, dx: {dx}, dy: {dy}, omega: {omega}')
                    self.set_move((max(min(Vy, 1), -1), 0, max(min(omega, 1), -1), -1))
                    while not self.depth_sensor.is_landed:
                        self.get_sensors_data()
                        time.sleep(0.01)
                    return 'aim'
                else:  # 坐底后考虑是否起跳调整位置
                    self.set_Vz(-1)
                    if self.target.roi_check():  # 检查位置阈值
                        self.aim_chances[0] = self.aim_chances[1]
                        print('🤿 AUV ready for grasping, start grasping!')
                        return 'grasp'
                    else:
                        print('🤿 👀 target not in range')
                        if abs(dx) > 1.5 * grasp_thresh_x:
                            omega = omega * 1.4
                        elif abs(dy) > grasp_thresh_y:
                            if abs(dy) > 1.5 * grasp_thresh_y:
                                Vy = dy * 1.6
                            else:
                                Vy = dy * 1.8
                            omega = 0
                        elif abs(dx) > grasp_thresh_x:
                            omega = omega * 1.1
                            Vy = 0
                        self.aim_chances[0] -= 1
                        print(f'🤿 target: {self.target.center}, dx: {dx}, dy: {dy}, omega: {omega}')
                        print(f'🤿 AUV try again! {self.aim_chances[0]} chances left')
                        self.set_move((max(min(Vy, 1), -1), 0, max(min(omega, 1), -1), 0.3))
                        time.sleep(0.5)  # 上浮0.5s
                        self.set_Vz(0)
                        return 'aim'
            else:
                # 放弃瞄准, 转坐底, 然后会转抓取, 如果还是没有东西会转巡航
                self.aim_chances[0] = self.aim_chances[1]
                print('🤿 AUV give up aiming, start landing')
                return 'land'

    def state_machine(self) -> str:
        cases = {
            'initial': self.initial,
            'aim': self.aim,
            'cruise': self.cruise,
            'grasp': self.grasp,
            'land': self.land,
        }
        self.state = cases[self.state]()
        print(f'🟢  {self.state}')
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
                if quit_flag:
                    screen.refresh()
                    if switch:
                        break
                    else:
                        switch = True
                # 更新target, arm数据
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as visual_socket:
                    try:
                        visual_socket.connect(('127.0.0.1', VISUAL_SERVER_PORT))
                    except ConnectionRefusedError:
                        print('[Visual Info Client] lost connection')
                        continue
                    # send flags to visual server
                    # threads_quit_flag: 2; arm_is_working: 1
                    visual_socket.send(bytes(str(quit_flag * 2).encode()))
                    # receive data from ROV then update target and arm
                    visual_info_dict = yaml.load(visual_socket.recv(1024), Loader=yaml.Loader)
                # quit_flag置1后待运行到下一循环, 将quit_flag发送给visual_server后再break
                # update AUV data
                auv.target.update(visual_info_dict["target"])
                auv.visual_arm.update(visual_info_dict["arm"])
                auv.get_sensors_data()  # 主要是获取深度
                # 状态机状态跳转并给出抓取判断
                grasp_state = auv.state_machine()
                # 软体臂控制
                if grasp_state == 'ready':
                    auv.visual_arm.arm_is_working = True
                    auv.visual_arm.start_time = time.time()
                elif grasp_state == 'activated':
                    auv.arm.controller.send((auv.target.center + (0,), auv.visual_arm.marker_position + (0,)))
                elif grasp_state == 'idle':
                    auv.visual_arm.arm_is_working = False

    Screen.wrapper(screen_main)
