#! /usr/bin/env python3

'''Soft manipulator driving module
provide pressure calculation functions with given end effector position
'''

import sys
from typing import Tuple, Generator
import time
import numpy as np
from scipy.optimize import root


class Manipulator():
    def __init__(self):
        self.d = 48.0  # unit: mm
        self.initBendLen = 110.0
        self.initElgLen = 125.0
        self.initZ = 2 * self.initBendLen + self.initElgLen
        self.segBendUpLen = [self.initBendLen] * 3
        self.segBendLowLen = [self.initBendLen] * 3
        self.segElgLen = self.initElgLen
        # the 10 channels are:
        # - upper bending segment chamber 1, 2, 3,
        # - lower bending segment chamber 1, 2, 3,
        # - elongation segment 1, 2, 3,
        # - hand
        self.pressures = [0.0] * 10
        self.water_pressure = 0.0
        self.bendPressureThresh = [0, 130]
        self.elgPressureThresh = [0, 30]
        self.handPressureThresh = [0, 60]
        self.controller = self.PID()
        self.controller.send(None)
        # create 10 channel pwm module instance
        if (__name__ == '__main__' and len(sys.argv) == 3 and sys.argv[2] == 'with_pwm') or __name__ != '__main__':
            from .pwm import PWM
            self.pwm = PWM()

    def set_pwm(self, pressures):
        # 5 channels used: 3 upper bending, 3 lower bending, 1 elongation, 1 hand
        p_list = pressures[0:7] + [pressures[-1]]
        for channel, p in enumerate(p_list):
            # 0-1 pwm duty cycle -> 0-5 V analog voltage -> 0-500 KPa pressure
            if channel == 3:
                # Jetson PWM模块的通道3坏了, 用通道10代替
                self.pwm.setValue(10, np.interp(p, [0, 500], [0, 1]))
                continue
            self.pwm.setValue(channel, np.interp(p, [0, 500], [0, 1]))

    def release(self):
        self.pwm.reset_all()
        self.segBendUpLen = [self.initBendLen] * 3
        self.segBendLowLen = [self.initBendLen] * 3
        self.segElgLen = self.initElgLen
        self.pressures = [0.0] * 10
        self.set_Pressures()
        self.controller = self.PID()
        self.controller.send(None)
        print('💪  Manipulator released 👌')

    def transform(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """transform vector from camera coordinate system to manipulator coordinate system

        Parameters
        ----------
        x : float
            x of end effector in camera coordinate system, unit: mm
        y : float
            y of end effector in camera coordinate system, unit: mm
        z : float
            z of end effector in camera coordinate system, unit: mm

        Returns
        -------
        Tuple[float, float, float]
            position of end effector in manipulator coordinate system
        """
        # TODO: 需要调参: 转移矩阵
        return x, y, z

    def inverse_kinematics(self, x: float, y: float, z: float) -> Tuple[Tuple[float, float, float], float]:
        """(Simplified algorithm for manual control) do inverse kinematics for the soft manipulator
        under the OBSS model with given position of end effector. In this inverse kinematics model,
        bending segments do not get involved in elongation actively.

        If end point in the workspace, set length of each chamber of two bending
        segments and one elongation segment; if not, do nothing

        Parameters
        ----------
        x : float
            x of end effector, unit: mm
        y : float
            y of end effector, unit: mm
        z : float
            z of end effector, unit: mm. Downward is the positive direction.
        """
        # determine phi based on x, y
        if x < 0:
            phi = -np.arctan(y / x) + np.pi
        elif x > 0:
            phi = -np.arctan(y / x)
        else:
            if y > 0:
                phi = -np.pi / 2
            elif y < 0:
                phi = np.pi / 2
            else:
                # when x, y are 0, lengths could be calculated easily
                self.segBendLowLen = (self.initBendLen, self.initBendLen, self.initBendLen)
                self.segElgLen = self.initElgLen + z - self.initZ
                return self.segBendLowLen, self.segElgLen
        # do phase shift and determine c to fit the function later
        if phi == -np.pi / 2 or phi == 1.5 * np.pi:
            # f(1) = (r - d * cos(phi + pi/6)) * theta - initLength;
            # f(2) = - r*(1-cos(theta))*sin(phi) - y;
            phiF = -phi + np.pi / 3
            phiP = phi + np.pi / 2
            c = y / 2
        elif -np.pi / 2 < phi < np.pi / 6:
            # f(1) = (r - d * cos(phi + pi/6)) * theta - initLength;
            # f(2) = r * cos(phi) * (1-cos(theta)) - x;
            phiF = -phi + np.pi / 3
            phiP = phi
            c = x / 2
        elif np.pi / 6 <= phi < np.pi * 5 / 6 and phi != np.pi / 2:
            # f(1) = (r - d * sin(phi)) * theta - initLength;
            # f(2) = r * cos(phi) * (1-cos(theta)) - x;
            phiF = phi
            phiP = phi
            c = x / 2
        elif phi == np.pi / 2:
            # f(1) = (r - d * sin(phi)) * theta - initLength;
            # f(2) = - r*(1-cos(theta))*sin(phi) - y;
            phiF = phi
            phiP = phi + np.pi / 2
            c = y / 2
        else:  # pi * 5 / 6 <= phi < pi * 3 / 2
            # f(1) = (r + d * cos(phi - pi/6)) * theta - initLength;
            # f(2) = r * cos(phi) * (1-cos(theta)) - x;
            phiF = phi - np.pi * 2 / 3
            phiP = phi
            c = x / 2
        # use Levenberg-Marquardt algorithm as Matlab's fsolve
        # scipy.optimize.root is better than scipy.optimize.fsolve. see:
        # https://stackoverflow.com/questions/21885093/comparing-fsolve-results-in-python-and-matlab
        solution = root(
            lambda x, phiF, phiP, c, d, len: [
                (x[0] - d * np.sin(phiF)) * x[1] - len,
                x[0] * np.cos(phiP) * (1 - np.cos(x[1])) - c
            ],
            [150, 0], args=(phiF, phiP, c, self.d, self.initBendLen), method='lm'
        )
        r, theta = solution.x
        segBendLen = (
            (r - self.d * np.sin(phi)) * theta,
            (r + self.d * np.cos(phi - np.pi / 6)) * theta,
            (r - self.d * np.cos(phi + np.pi / 6)) * theta
        )
        segElgLen = -r * np.sin(theta) * 2 + z
        return segBendLen, segElgLen

    def len2pressures(self, segLen: Tuple[Tuple[float, float, float], float], pressures_dict=None) -> bool:
        """convert length of chambers to pressures of chambers
        Parameters
        ----------
        segLen: Tuple[Tuple[float, float, float], float]
            tuple of segBendLen and segElgLen. segBendLen is tuple of length of
            three chambers in the bending segment, segElgLen is length of three
            chambers in the elongation segment
        pressures_dict : dict, optional
            a dictionary of pressures of specified chambers. key is int (0~9),
            value is float. e.g. {9: 90} to set pressure of hand, by default None

        Returns
        -------
        is_in_workspace: give bool value of whetcher the end point is in workspace
            based on the pressure thresholds
        """
        segBendLen, segElgLen = segLen
        # arguments from calibration with initial length of bending chambers: 110
        # initial length of elongation chambers: 125
        a = 5555.5555555555555555555555555556
        b = 602224.58212670832698267540517198
        c = 93258097.726418903983160271113075
        d = 453.48422496570644718792866941015
        e = 4.8148148148148148148148148148148
        pressures = [0.0] * 10
        # lower bending segment
        pressures[0:3] = map(
            lambda l: (
                (a * l + ((a * l - b)**2 + c) ** 0.5 - b)**(1 / 3) +
                - d / (a * l + ((a * l - b)**2 + c)**0.5 - b)**(1 / 3) - e
            ),
            segBendLen
        )
        # upper bending segment
        # 稍微增大上段气压以平衡重力影响
        pressures[3:6] = map(lambda p: 0.98 * p - 0.6413, pressures[0:3])
        # elongation segment
        pressures[6:9] = [0.51789321 * segElgLen - 64.06856906] * 3
        # cover pressures specified manually
        if pressures_dict is not None:
            for key in pressures_dict:
                pressures[key] = pressures_dict[key]
        self.pressures = pressures.copy()
        self.segBendLowLen, self.segBendUpLen, self.segElgLen = segBendLen, segBendLen, segElgLen
        return self.set_Pressures()

    def set_Pressures(self) -> bool:
        """set pressures of all chambers then set pwm value of all channels.
        could set pressure of hand with pressures_dict

        Returns
        -------
        is_in_workspace: give bool value of whetcher the end point is in workspace
            based on the pressure thresholds
        """
        pressures = self.pressures.copy()
        # decide whether end point is in workspace by pressure threshold
        if not (
            all([self.bendPressureThresh[0] <= p <= self.bendPressureThresh[1] for p in pressures[0:3]])
            and self.elgPressureThresh[0] <= pressures[6] <= self.elgPressureThresh[1]
        ):
            print(f'💪  ❌ exceed pressure threshold! segBendUp:', ', '.join(f'{p:.3f}' for p in pressures[0:3]), f'segElg: {pressures[6]:.3f}')
            return False
        # 当气压值小于6气动阀输出不稳定, 因此直接置零
        for i, p in enumerate(pressures):
            if p <= 6:
                pressures[i] = 0
        pressures[3:6] = np.clip(pressures[3:6], self.bendPressureThresh[0], self.bendPressureThresh[1])
        pressures[-1] = np.clip(pressures[-1], self.handPressureThresh[0], self.handPressureThresh[1])
        # balance the influence of outside water pressure
        pressures = [p + self.water_pressure for p in pressures]
        print(f'💪  💨 pressures:', ', '.join(f'{p:.3f}' for p in pressures), f'🌊 water pressure: {self.water_pressure:.3f}')
        if (__name__ == '__main__' and len(sys.argv) == 3 and sys.argv[2] == 'with_pwm') or __name__ != '__main__':
            self.set_pwm(pressures)
        return True

    def reset(self):
        """reset manipulator to initial position
        """
        self.pwm.reset_all()
        self.pressures = [0, 120, 100] + [0] * 7
        self.set_Pressures()
        self.hand('close')
        print('💪 Arm reset')

    def fold(self, is_on: bool):
        """fold elongation segment
        """
        if is_on:
            self.pwm.setValue(8, 0.99)
            self.pressures[6:9] = [0, 0, 0]
            self.set_Pressures()
        else:
            self.pwm.setValue(8, 0)

    def collect(self):
        """collect grasped target into basket
        """
        # 甩回来
        self.len2pressures(self.inverse_kinematics(0, 0, self.initZ), pressures_dict={1: 120, 2: 120, 3: 100, 5: 100, 9: 40})
        time.sleep(2)  # wait for 2s
        # 放进去
        self.pwm.setValue(8, 0)
        self.len2pressures(self.inverse_kinematics(0, 0, self.initZ), pressures_dict={1: 120, 2: 120, 3: 100, 5: 100, 6: 30, 7: 30, 8: 30, 9: 40})
        time.sleep(4)  # wait for 4s
        # 松手
        self.len2pressures(self.inverse_kinematics(0, 0, self.initZ), pressures_dict={1: 120, 2: 120, 3: 100, 5: 100, 6: 30, 7: 30, 8: 30})
        self.pwm.setValue(9, 0.90)
        time.sleep(2)
        # 收伸长段
        self.len2pressures(self.inverse_kinematics(0, 0, self.initZ), pressures_dict={1: 120, 2: 120, 3: 100, 5: 100})
        self.fold(True)
        time.sleep(1)
        # 归位
        self.reset()
        print('💪 Arm collecting')

    def hand(self, mode: str):
        """set hand to specific mode

        Parameters
        ----------
        mode : str
            open/close/idle
        """
        if mode == 'open':
            self.pwm.setValue(9, 0.90)
            self.pressures[-1] = 0
            print('🖐 hand opened')
        elif mode == 'close':
            self.pwm.setValue(9, 0)
            self.pressures[-1] = 60
            print('🤌 hand closed')
        elif mode == 'idle':
            self.pwm.setValue(9, 0)
            self.pressures[-1] = 0
        self.set_Pressures()

    def PID(self) -> Generator[None, Tuple[Tuple[float, float, float], Tuple[float, float, float]], None]:
        """simple closed loop feedback on error of arm position and target position

        For instruction on calibrate PID, see: https://zh.wikipedia.org/wiki/齐格勒－尼科尔斯方法

        Parameters
        ----------
        target_pos : Tuple[float, float, float]
            tuple of target end point position, [x, y, z] in mm
        arm_pos : Tuple[float, float, float]
            tuple of arm end point position, [x, y, z] in mm

        Returns
        -------
        bool
            if False, feedback exceed system limit
        """
        # TODO: 需要调参: PID系数
        Kp = np.array([5, 5, 5])  # x, y, z
        Ki = np.array([5, 5, 5])  # x, y, z
        Kd = np.array([0, 0, 0])  # x, y, z
        e_prev = np.zeros(3)
        t_prev = time.time()
        integral = np.zeros(3)
        self.state = False
        while True:
            target_pos, arm_pos = yield
            print(f'target: {target_pos[:3]}, arm: {arm_pos[:3]}')
            t = time.time()
            e = np.multiply(np.array(target_pos) - np.array(arm_pos), np.array([1000, 800, 0]))
            abs_e = sqrt(e[0] ** 2 + e[1] ** 2)
            if abs_e < 10:
                self.state = True
            proportional = np.multiply(Kp, e)
            integral = integral + np.multiply(Ki, e) * (t - t_prev)
            derivative = np.multiply(Kd, e - e_prev) / (t - t_prev)
            x_corrected, y_corrected, z_corrected = tuple(proportional + integral + derivative)
            print(x_corrected, y_corrected, self.initZ + 25, abs_e)
            # self.len2pressures(self.inverse_kinematics(x_corrected, y_corrected, z_corrected))
            self.len2pressures(self.inverse_kinematics(x_corrected, y_corrected, self.initZ + 25))
            # TODO: 需要调参: 采样时间
            time.sleep(1)  # sleep for 1s


if __name__ == "__main__" and sys.argv[1] == 'auto':
    """run one of following command
      >>> python ./manipulate.py auto
      >>> python ./manipulate.py auto with_pwm
    """
    arm = Manipulator()
    target_pos = (0, 0, 355)
    # start the PID controller
    arm.controller.send(None)
    for i in range(50):
        arm_pos = (-1, -1, 355)
        arm.controller.send((target_pos, arm_pos))

if __name__ == '__main__' and sys.argv[1] == 'manual':
    """run one of following command
      >>> python ./manipulate.py manual
      >>> python ./manipulate.py manual with_pwm
    """
    arm = Manipulator()
    arm.water_pressure = float(input('input water pressure: '))
    print('- input format: [x],[y],[z]\n- input q to quit')
    while True:
        command = input('input: ')
        if command == 'q':
            break
        elif command == '':
            continue
        arm.len2pressures(eval(f'arm.inverse_kinematics({command})'))
