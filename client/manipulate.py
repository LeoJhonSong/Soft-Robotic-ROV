#! /usr/bin/env python3

'''Soft manipulator driving module
provide pressure calculation functions with given end effector position
'''

import sys
from typing import Tuple
from time import sleep
import numpy as np
from numpy import arcsin, arctan, ceil, cos, dot, exp, pi, sign, sin, sqrt
from scipy.optimize import root
from scipy.signal import cont2discrete


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
        # create 10 channel pwm module instance
        if (__name__ == '__main__' and len(sys.argv) == 3 and sys.argv[2] == 'with_pwm') or __name__ != '__main__':
            from pwm import PWM
            self.pwm = PWM()

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
        under the OBSS model with given position of end effector.

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
            phi = -arctan(y / x) + pi
        elif x > 0:
            phi = -arctan(y / x)
        else:
            if y > 0:
                phi = -pi / 2
            elif y < 0:
                phi = pi / 2
            else:
                # when x, y are 0, lengths could be calculated easily
                self.segBendLowLen = (self.initBendLen, self.initBendLen, self.initBendLen)
                self.segElgLen = self.initElgLen + z - self.initZ
                return self.segBendLowLen, self.segElgLen
        # do phase shift and determine c to fit the function later
        if phi == -pi / 2 or phi == 1.5 * pi:
            # f(1) = (r - d * cos(phi + pi/6)) * theta - initLength;
            # f(2) = - r*(1-cos(theta))*sin(phi) - y;
            phiF = -phi + pi / 3
            phiP = phi + pi / 2
            c = y / 2
        elif -pi / 2 < phi < pi / 6:
            # f(1) = (r - d * cos(phi + pi/6)) * theta - initLength;
            # f(2) = r * cos(phi) * (1-cos(theta)) - x;
            phiF = -phi + pi / 3
            phiP = phi
            c = x / 2
        elif pi / 6 <= phi < pi * 5 / 6 and phi != pi / 2:
            # f(1) = (r - d * sin(phi)) * theta - initLength;
            # f(2) = r * cos(phi) * (1-cos(theta)) - x;
            phiF = phi
            phiP = phi
            c = x / 2
        elif phi == pi / 2:
            # f(1) = (r - d * sin(phi)) * theta - initLength;
            # f(2) = - r*(1-cos(theta))*sin(phi) - y;
            phiF = phi
            phiP = phi + pi / 2
            c = y / 2
        else:  # pi * 5 / 6 <= phi < pi * 3 / 2
            # f(1) = (r + d * cos(phi - pi/6)) * theta - initLength;
            # f(2) = r * cos(phi) * (1-cos(theta)) - x;
            phiF = phi - pi * 2 / 3
            phiP = phi
            c = x / 2
        # use Levenberg-Marquardt algorithm as Matlab's fsolve
        # scipy.optimize.root is better than scipy.optimize.fsolve. see:
        # https://stackoverflow.com/questions/21885093/comparing-fsolve-results-in-python-and-matlab
        solution = root(
            lambda x, phiF, phiP, c, d, len: [
                (x[0] - d * sin(phiF)) * x[1] - len,
                x[0] * cos(phiP) * (1 - cos(x[1])) - c
            ],
            [150, 0], args=(phiF, phiP, c, self.d, self.initBendLen), method='lm'
        )
        r, theta = solution.x
        segBendLen = (
            (r - self.d * sin(phi)) * theta,
            (r + self.d * cos(phi - pi / 6)) * theta,
            (r - self.d * cos(phi + pi / 6)) * theta
        )
        segElgLen = -r * sin(theta) * 2 + z
        return segBendLen, segElgLen

    def set_pwm(self):
        # 5 channels used: 3 bending, 1 elongation, 1 hand
        p_list = self.pressures[0:3] + self.pressures[8:]
        for channel, p in enumerate(p_list):
            # 0-1 pwm duty cycle -> 0-5 V analog voltage -> 0-500 KPa pressure
            self.pwm.setValue(channel, np.interp(p, [0, 500], [0, 1]))

    def set_Pressures(self, segLen: Tuple[Tuple[float, float, float], float], pressures_dict=None):
        """set pressures of chambers `in the arm` (hand pressure not set here)
        then set pwm value of all channels. set pressure of hand with pressures_dict

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
        pressures[3:6] = map(
            lambda l: (
                (a * l + ((a * l - b)**2 + c) ** 0.5 - b)**(1 / 3) +
                - d / (a * l + ((a * l - b)**2 + c)**0.5 - b)**(1 / 3) - e
            ),
            segBendLen
        )
        # upper bending segment
        # 稍微减小上段气压以平衡重力影响
        pressures[0:3] = map(lambda p: 0.98 * p - 0.6413, pressures[3:6])
        # elongation segment
        pressures[6:9] = [0.51789321 * segElgLen - 64.06856906] * 3
        # cover pressures specified manually
        if pressures_dict is not None:
            for key in pressures_dict:
                pressures[key] = pressures_dict[key]
        # decide whether end point is in workspace by pressure threshold
        if not (
            all([self.bendPressureThresh[0] <= p <= self.bendPressureThresh[1] for p in pressures[3:6]])
            and self.elgPressureThresh[0] <= pressures[6] <= self.elgPressureThresh[1]
        ):
            print(f'[Arm] ❌ exceed pressure threshold! segBendLow:', ', '.join(f'{p:.3f}' for p in pressures[3:6]), f'segElg: {pressures[6]:.3f}')
            return False
        self.segBendLowLen, self.segBendUpLen, self.segElgLen = segBendLen, segBendLen, segElgLen
        pressures[0:3] = np.clip(pressures[0:3], self.bendPressureThresh[0], self.bendPressureThresh[1])
        pressures[-1] = np.clip(pressures[-1], self.handPressureThresh[0], self.handPressureThresh[1])
        # balance the influence of outside water pressure
        self.pressures = [p + self.water_pressure for p in pressures]
        print(f'[Arm] 💨 pressures:', ', '.join(f'{p:.3f}' for p in self.pressures), f'🌊 water pressure: {self.water_pressure:.3f}')
        if (__name__ == '__main__' and len(sys.argv) == 3 and sys.argv[2] == 'with_pwm') or __name__ != '__main__':
            self.set_pwm()
        return True

    def closed_loop_move(self, target_pos: Tuple[float, float, float], arm_pos: Tuple[float, float, float]):
        """simple closed loop feedback on error of arm position and target position

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
        # TODO: change to PID
        kp = 1  # TODO: 需要调参: 比例系数
        x_corrected, y_corrected, z_corrected = tuple(map(lambda a, t: t - kp * (a - t), arm_pos, target_pos))
        x_corrected, y_corrected, z_corrected = self.transform(x_corrected, y_corrected, z_corrected)
        return self.set_Pressures(self.inverse_kinematics(x_corrected, y_corrected, z_corrected))


if __name__ == "__main__" and sys.argv[1] == 'auto':
    """run one of following command
      >>> python ./manipulate.py auto
      >>> python ./manipulate.py auto with_pwm
    """
    arm = Manipulator()
    target_pos = (0, 0, 345)
    while True:
        arm_pos = (-1, -1, 345)
        arm.closed_loop_move(target_pos, arm_pos)

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
        arm.set_Pressures(eval(f'arm.inverse_kinematics({command})'))
