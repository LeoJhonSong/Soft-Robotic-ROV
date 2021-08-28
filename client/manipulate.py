#! /usr/bin/env python3

'''Soft manipulator driving module
provide pressure calculation functions with given end effector position
'''

import sys
from typing import Union
from time import sleep
import numpy as np
from numpy import arcsin, arctan, ceil, cos, dot, exp, pi, sign, sin, sqrt
from scipy.optimize import root
from scipy.signal import cont2discrete


class Chamber():
    def __init__(self) -> None:
        """a single chamber in a segment of the soft manipulator
        """
        # RLKMIC奖惩值表
        self.RA = np.array([
            [0, -1, -1, -1, -1, -1, -1],
            [0, 0, -1, -1, -1, -1, -1],
            [1, 1, 1, 0, 0, 0, 0],
            [10, 10, 10, 10, 10, 10, 10],
            [0, 0, 0, 0, 1, 1, 1],
            [-1, -1, -1, -1, -1, 0, 0],
            [-1, -1, -1, -1, -1, 0, 0]
        ])
        self.alpha = 1  # 学习因子
        self.gama = 0.8  # 折扣因子
        # PID用的东西
        self.state_next_PID = 0
        self.action_next_PID = 0
        self.Q_PID = np.zeros([7, 7])  # PID用的Q值表
        self.kp, self.ki, self.kd = 1, 0.001, 0.01
        # A, B, C, D for Linear Time Invariant system in state-space form
        self.A = np.array([
            [-self.kp / self.kd, -self.ki / self.kd],
            [1, 0]
        ])
        self.B = np.array([[1, 0]]).T
        self.C = np.array([-1 / self.kd, 0])
        self.D = 107  # although set to the initial length, this value does not matter
        # Kalman用的东西
        self.Q_Kal = 1
        self.R_Kal = 1
        self.P_Kal = self.B * self.Q_Kal * self.B.T
        self.x_Kal = np.zeros([2, 1])
        self.pp_Kal = [0]
        # MIC2用的东西
        self.state_MIC = 3
        self.action_MIC = [i for i in range(7)]
        self.Q_MIC = np.zeros([7, 7, 7])  # MIC用的值函数矩阵Q
        self.ier = 0
        # 三个算法共用的变量
        self.sz = 0
        self.action = 0
        self.lamda = 0

    def predictor(self, x):
        response = None
        return response

    def kringpredict(self, len_list):
        """Preprocessing func for predictor

        Parameters
        ----------
        len_list : numpy array of four array of length

        Returns
        -------
        float
            predicted responce at len_list
        """
        diff10 = len_list[1] - len_list[0]
        diff21 = len_list[2] - len_list[1]
        diff32 = len_list[3] - len_list[2]
        x = np.vstack((
            len_list[1],
            len_list[2],
            diff21,
            (diff10 == diff21) & (diff21 == diff32)
        )).T
        return self.predictor(x)

    def kalman(self, len_now, len_target, pressure):
        self.D = len_target
        Ad, Bd, Cd, Dd, _ = cont2discrete([self.A, self.B, self.C, self.D], 0.01)
        P = dot(dot(Ad, self.P_Kal), Ad.T) + dot(dot(Bd, self.Q_Kal), Bd.T)  # matrix
        Mn = dot(P, Cd.T)[:, np.newaxis] / (dot(dot(Cd, P), Cd.T) + self.R_Kal)  # column vector
        P = dot(np.eye(2) - Mn * Cd, P)
        x_now = self.x_Kal  # column vector
        x_now = dot(Ad, x_now) + Bd * pressure
        x_predict = x_now + (len_now - dot(Cd, x_now) - Dd) * Mn  # column vector
        len_predict = float(dot(Cd, x_predict)) + Dd  # number
        p1 = self.D - len_now
        p2 = self.D - len_predict
        self.pp_Kal = np.append(self.pp_Kal, abs(p1))
        p3 = self.pp_Kal.max()
        if (
            p1 < -0.001 and 0 < p2 < ceil(2 * p3)
        ) or (
            p1 > 0.001 and -ceil(2 * p3) < p2 < 0
        ) or (
            np.sign(p1) == np.sign(p2) and abs(p1) - abs(p2) > ceil(p3)
        ):
            len_predict = float(dot(Cd, x_predict)) + Dd - ceil(p1)
            sz = -self.D + len_predict
        else:
            sz = float(dot(dot(Cd, P), Cd.T))  # number
        # return x_next, y_next, sz, P
        self.x_Kal, self.sz, self.P_Kal = x_predict, sz, P
        return len_predict

    def GRLKPID(self):
        sz, kp, R, Q = -self.sz, self.kp, self.R_Kal, self.Q_Kal
        state_now, action_now, Q_table = self.state_next_PID, self.action_next_PID, self.Q_PID
        # 贪婪算法动作选择策略
        action_next = 0
        state_next = 0
        action_ranges = [-np.inf, -50, -1, -0.0001, 0.0001, 1, 50, np.inf]
        if True:
            for i in range(len(action_ranges) - 1):
                if action_ranges[i] <= sz < action_ranges[i + 1]:
                    if i == 0:
                        action_next = 0
                    elif i == 6:
                        action_next = 6
                    elif i == 3:
                        action_next = np.argmax(Q_table[3, 2:5])
                    else:
                        action_next = np.argmax(Q_table[i, i - 1:i + 1])
                    if i in [2, 3]:
                        action_next += i - 1
                    elif i in [4, 5]:
                        action_next += i
                    state_next = i

        # 更新Q值表
        Q_table[state_now, action_now] += self.alpha * (
            self.RA[state_next, action_next] +
            + self.gama * Q_table[state_next, action_next] - Q_table[state_now, action_now]
        )
        # 动作集
        action_dict = {
            0: (
                kp + (0.2 * exp(1 - 50 / abs(sz)) + 0.02),
                R + (0.1 * exp(1 - 50 / abs(sz)) + 0.01),
                Q - 0.1 * exp(1 - 50 / abs(sz)) - 0.01
            ),
            1: (
                kp + (0.02 / exp(1 - 1 / 50) * exp(1 - 1 / abs(sz)) + 0.002),
                R + (0.01 / exp(1 - 1 / 50) * exp(1 - 1 / abs(sz)) + 0.001),
                Q - (0.01 / exp(1 - 1 / 50) * exp(1 - 1 / abs(sz)) + 0.001)
            ),
            2: (
                kp + (0.002 / exp(1 - 0.001) * exp(1 - 0.0001 / abs(sz)) + 0.0002),
                R + (0.001 / exp(1 - 0.0001) * exp(1 - 0.0001 / abs(sz)) + 0.0001),
                Q - (0.001 / exp(1 - 0.0001) * exp(1 - 0.0001 / abs(sz)) + 0.0001)
            ),
            3: (
                kp - (0.0002 * exp(1 - 0.0001 / abs(sz))) * sign(sz),
                R - (0.0001 * exp(1 - 0.0001 / abs(sz))) * sign(sz),
                Q + (0.0001 * exp(1 - 0.0001 / abs(sz))) * sign(sz)
            ),
            4: (
                kp - (0.002 / exp(1 - 0.0001) * exp(1 - 0.0001 / abs(sz)) + 0.0002),
                R - (0.001 / exp(1 - 0.0001) * exp(1 - 0.0001 / abs(sz)) + 0.0001),
                Q + (0.001 / exp(1 - 0.0001) * exp(1 - 0.0001 / abs(sz)) + 0.0001)
            ),
            5: (
                kp - (0.02 / exp(1 - 1 / 50) * exp(1 - 1 / abs(sz)) + 0.0002),
                R - (0.01 / exp(1 - 1 / 50) + exp(1 - 1 / abs(sz)) + 0.001),
                Q + (0.01 / exp(1 - 1 / 50) * exp(1 - 1 / abs(sz)) + 0.001)
            ),
            6: (
                kp - 0.2 * exp(1 - 50 / abs(sz)) - 0.02,
                R - 0.1 * exp(1 - 50 / abs(sz)) - 0.01,
                Q + 0.1 * exp(1 - 50 / abs(sz)) + 0.01
            )
        }
        # 根据action_next选择公式更新kp, R, Q
        kp, R, Q = action_dict[np.clip(action_next, 0, 6)]
        kp = np.clip(kp, 0.00001, np.inf)
        # return kp, R, Q, state_next, action_next, Q_table
        self.kp, self.R_Kal, self.Q_Kal, self.state_next_PID, self.action_next_PID, self.Q_PID = kp, R, Q, state_next, action_next, Q_table
        return action_now

    def RLKMIC2(self, len_target, len_predict, action_PID_now):
        lamda, sz, Q_func_table, state_now, action_now = self.lamda, self.sz, self.Q_MIC, self.state_MIC, self.action_MIC
        action_PID_next, ier = self.action_next_PID, self.ier
        eer = len_target - len_predict
        ier += eer * 0.01
        s = -sqrt((eer**2 + sz**2) / 2) * sign(-eer + sz)
        if np.isnan(s):
            s = 0.000001
        # 根据贪婪策略选择动作
        action_ranges = [-np.inf, -50, -1, -0.00001, 0.00001, 1, 50, np.inf]
        state_next = 0
        action_next = 0
        if True:
            for i in range(len(action_ranges) - 1):
                if action_ranges[i] <= s < action_ranges[i + 1]:
                    if i == 0:
                        action_next = 0
                    elif i == 6:
                        action_next = 6
                    elif i == 3:
                        Qk = Q_func_table[action_PID_now, 3, 2:5]
                        index_ruler = np.array([n for n in range(len(Qk) - 1, -1, -1)])
                        action_next = index_ruler[np.argsort(Qk[::-1])][0]
                    else:
                        Qk = Q_func_table[action_PID_now, i, i - 1:i + 1]
                        index_ruler = np.array([n for n in range(len(Qk) - 1, -1, -1)])
                        action_next = index_ruler[np.argsort(Qk[::-1])][0]
                    if i in [2, 3]:
                        action_next += i - 1
                    elif i in [4, 5]:
                        action_next += i
                    state_next = i
        # 更新值函数表
        if isinstance(action_now, list):
            Q_func_table[action_PID_now, state_now, action_now[0]:action_now[-1] + 1] += self.alpha * (
                self.RA[state_next, action_now[0]:action_now[-1] + 1] +
                + self.gama * Q_func_table[action_PID_next, state_next, action_next] +
                - Q_func_table[action_PID_now, state_next, action_now[0]:action_now[-1] + 1]
            )
        else:
            Q_func_table[action_PID_now, state_now, action_now] += self.alpha * (
                self.RA[state_next, action_now] +
                + self.gama * Q_func_table[action_PID_next, state_next, action_next] +
                - Q_func_table[action_PID_now, state_next, action_now]
            )
        state_now = state_next
        action_now = action_next
        action_dict = {
            0: lamda - (10 * exp(1 - 50 / abs(s)) + 1),
            1: lamda - (0.1 / exp(1 - 1 / 50) * exp(1 - 1 / abs(s)) + 0.01),
            2: lamda - (0.01 / exp(1 - 0.00001) * exp(1 - 0.0001 / abs(s)) + 0.001),
            3: lamda - (0.0001 * exp(1 - 0.00001 / abs(s))) * sign(s),
            4: lamda + (0.01 / exp(1 - 0.00001) * exp(1 - 0.0001 / abs(s)) + 0.001),
            5: lamda + (0.1 / exp(1 - 1 / 50) * exp(1 - 1 / abs(s)) + 0.01),
            6: lamda + (10 * exp(1 - 50 / abs(s)) + 1)
        }
        # 根据action_next值选择公式更新lambda
        lamda = action_dict[np.clip(action_now, 0, 6)]
        # return lamda, Q_func_table, ier, state_now, action
        self.lamda, self.Q_MIC, self.ier, self.state_MIC, self.action_MIC = lamda, Q_func_table, ier, state_now, action_now


class Manipulator():
    def __init__(self) -> None:
        self.d = 48.0  # unit: mm
        self.initBendLen = 107.0
        self.initElgLen = 102.0
        self.maxBendLen = 200.0
        self.maxElgLen = 170.0
        self.segBendUpLen = [0.0] * 3
        self.segBendLowLen = [0.0] * 3
        self.segElgLen = 0.0
        # the 10 channels are:
        # - upper bending segment chamber 1, 2, 3,
        # - lower bending segment chamber 1, 2, 3,
        # - elongation segment 1, 2, 3,
        # - hand
        self.pressures = [0.0] * 10
        self.water_pressure = 0.0
        self.arm_position_resize = [2, 0.7]  # x, y  # TODO: 需要调参: 位置系数
        self.chambers = [Chamber() for i in range(7)]  # Upper1, 2, 3, Lower1, 2, 3, Elongation
        # create 10 channel pwm module instance
        if (__name__ == '__main__' and sys.argv[2] == 'with_pwm') or __name__ != '__main__':
            from pwm import PWM
            self.pwm = PWM()

    def inverse_kinematics_simplified(self, x: float, y: float, z: float) -> bool:
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
            z of end effector, upward is the positive direction. initial z is -325

        Returns
        -------
        is_in_workspace: give bool value of whetcher the end point is in workspace
        """
        initZ = -325.0  # TODO: 需要调参: 初始长度
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
                if z >= 0:
                    self.segBendLowLen = [self.initBendLen] * 3
                    self.segElgLen = self.initElgLen + z
                    return True
                else:
                    return False
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
        segBendLen = [
            (r - self.d * sin(phi)) * theta,
            (r + self.d * cos(phi - pi / 6)) * theta,
            (r - self.d * cos(phi + pi / 6)) * theta
        ]
        segElgLen = -r * sin(theta) * 2 - (z + initZ)
        # print(f'bend: {segBendLen}, elg: {segElgLen}')  # DEBUG
        # check if end point in workspace, if so, set the lengths
        # TODO: change to limit pressure
        if (
            all(self.initBendLen < length < self.maxBendLen for length in segBendLen)
            and self.initElgLen < segElgLen < self.maxElgLen
        ):
            # length of chamber 1, 2, 3 of upper bending segments
            self.segBendUpLen = segBendLen
            # length of chamber 1, 2, 3 of lower bending segments
            self.segBendLowLen = segBendLen
            # length of all three chambers of the elongation segment
            self.segElgLen = segElgLen
            return True
        else:
            return False

    def set_pwm(self):
        # 5 channels used: 3 bending, 1 elongation, 1 hand
        p_list = self.pressures[0:3] + self.pressures[8:]
        for channel, p in enumerate(p_list):
            # 0-1 pwm duty cycle -> 0-5 V analog voltage -> 0-500 KPa pressure
            self.pwm.setValue(channel, np.interp(p, [0, 500], [0, 1]))

    def set_Pressures(self, lamdas=None, pressures_dict=None):
        """set pressures of chambers `in the arm` (hand pressure not set here)
        then set pwm value of all channels. set pressure of hand with pressures_dict

        Parameters
        ----------
        lamdas : list, optional
            [description], by default None
        pressures_dict : dict, optional
            a dictionary of pressures of specified chambers. key is int (0~9),
            value is float. e.g. {9: 90} to set pressure of hand, by default None
        """
        # arguments from calibration
        a = 5555.5555555555555555555555555556
        b = 602224.58212670832698267540517198
        c = 93258097.726418903983160271113075
        d = 453.48422496570644718792866941015
        e = 4.8148148148148148148148148148148
        # lower bending segment
        self.pressures[3:6] = map(
            lambda l: (
                (a * l + ((a * l - b)**2 + c) ** 0.5 - b)**(1 / 3) +
                - d / (a * l + ((a * l - b)**2 + c)**0.5 - b)**(1 / 3) - e
            ),
            self.segBendUpLen
        )
        # upper bending segment
        self.pressures[0:3] = self.pressures[3:6]
        # 稍微减小上段气压以平衡重力影响
        # self.pressures[0:3] = map(lambda p: 0.98 * p - 0.6413, self.pressures[3:6])
        # elongation segment
        self.pressures[6:9] = [0.51789321 * self.segElgLen - 64.06856906] * 3
        pressures_to_send = self.pressures.copy()
        if lamdas is not None:
            for i in range(0, 3):
                pressures_to_send[i] = 1.5 * pressures_to_send[i] + 1 * lamdas[i]
            for i in range(3, 6):
                pressures_to_send[i] = pressures_to_send[i - 3] + 0.01 * lamdas[i]
        else:
            pressures_to_send[0:3] = [0.98 * p - 0.6413 for p in pressures_to_send[3:6]]
        # cover pressures specified manually
        if pressures_dict is not None:
            for key in pressures_dict:
                pressures_to_send[key] = pressures_dict[key]
        pressures_to_send[0:6] = np.clip(pressures_to_send[0:6], 0, 130)
        pressures_to_send[6:9] = np.clip(pressures_to_send[6:9], 0, 30)
        pressures_to_send[-1] = np.clip(pressures_to_send[-1], 0, 60)
        print(f'[Arm] 💨 pressures: {pressures_to_send}')
        if (__name__ == '__main__' and sys.argv[2] == 'with_pwm') or __name__ != '__main__':
            self.set_pwm()

    def inverse_kinematics(self, x: float, y: float, z: float, segElgLen=170) -> Union[list, None]:
        """(Alternate algorithm for auto grasping) Do inverse kinematics for the
        soft manipulator under the OBSS model with given position of end
        effector.

        If end point in the workspace, set length of each chamber of two bending
        segments and one elongation segment; if not, do nothing

        Parameters
        ----------
        x : float
            x of end effector, unit: mm
        y : float
            y of end effector, unit: mm
        z : float
            z of end effector, unit: mm. upward is the positive direction, which
            means z should always by `negative`
        segElgLen : float
            the length of chambers of the elongation segment, by default 170. # TODO: 需要调参: 伸长段长度

        Returns
        -------
        is_in_workspace: give bool value of whetcher the end point is in workspace
        """
        z_segBend = abs(z) - segElgLen  # 两个弯曲段在z轴上的长度
        # 弯曲段1 (Upper) 的偏转角theta1
        if x == 0:
            theta1 = sign(y) * pi / 2
        elif x > 0 and y >= 0:
            theta1 = arctan(y / x)
        elif x > 0 and y < 0:
            theta1 = arctan(y / x) + 2 * pi
        else:  # x < 0
            theta1 = arctan(y / x) + pi
        # 弯曲段2 (Lower) 的偏转角theta2
        theta2 = theta1 + pi
        # 弯曲段1, 2向心角phi
        phi = pi - 2 * arcsin(z_segBend / (x**2 + y**2 + z_segBend**2)**0.5)
        # 弯曲段1, 2曲率半径r
        r = ((x**2 + y**2 + z_segBend**2) / (8 * (1 - cos(phi))))**0.5
        segBendUpLen = [
            phi * (r - self.d / 2 * cos(theta1)),
            phi * (r - self.d / 2 * cos(2 / 3 * pi - theta1)),
            phi * (r - self.d / 2 * cos(4 / 3 * pi - theta1))
        ]
        segBendLowLen = [
            abs(phi) * (r - self.d / 2 * cos(theta2)),
            abs(phi) * (r - self.d / 2 * cos(2 / 3 * pi - theta2)),
            abs(phi) * (r - self.d / 2 * cos(4 / 3 * pi - theta2))
        ]
        # check if solution in workspace
        if (
            all(self.initBendLen <= length <= self.maxBendLen for length in segBendUpLen + segBendLowLen)
            and self.initElgLen <= segElgLen <= self.maxElgLen
        ):
            return segBendUpLen + segBendLowLen + [segElgLen]
        # it means the end point is out of workspace
        else:
            return None

    def route_gen(self, x: float, y: float):
        """Generate route (actually only one step) for reaching point on seabed
        at x, y. Therefore is actually to set segBendUpLen, segBendLowLen and
        segElgLen

        如果俯视手臂, 向后为x正方向, 向右为y正方向
        """
        # 因为坐标系的问题也可能要x, y互换
        x = self.arm_position_resize[0] * x
        y = self.arm_position_resize[1] * y
        z = 400.0  # TODO: 需要调参: 软体臂顶端距海床大致距离
        lens = self.inverse_kinematics(x, y, -z, segElgLen=z * 1 / 3)  # FIXME: is this specific segElgLen needed?
        if lens is not None:
            self.segBendUpLen, self.segBendLowLen, self.segElgLen = lens[0:3], lens[3:6], lens[-1]
        else:
            print('target out of workspace')

    def RLKMIC(self):
        """RLK: reinforce learning kalman
        """
        # 因为手臂运动过程中标记可能被遮挡所以目前无法实时更新
        x_target, y_target, z_target = 20, 20, 400
        lens_target = self.inverse_kinematics(self.arm_position_resize[0] * x_target, self.arm_position_resize[1] * y_target, -z_target)
        if lens_target is None:
            print('[Arm] target out of workspace')
            return
        steps = 30
        for step in range(steps):
            print(f'🟢 {step}:', end='\n    ')
            lamdas_list = [chamber.lamda for chamber in self.chambers[0:6]]
            if step > steps - 3:
                pressures = {6: 20, 7: 20, 8: 20}
                self.set_Pressures(lamdas=lamdas_list, pressures_dict=pressures)
            else:
                pressures = {6: 0, 7: 0, 8: 0}
                self.set_Pressures(lamdas=lamdas_list, pressures_dict=pressures)
            if (__name__ == '__main__' and sys.argv[2] == 'with_pwm') or __name__ != '__main__':
                sleep(1)  # wait for 1s to apply the pressures
            # TODO: 接收目标, 手爪位置更新
            x_hand, y_hand = input().strip().split(',')
            x_hand, y_hand, z_hand = float(x_hand), float(y_hand), 400
            print(f'    x: {x_hand}, y: {y_hand}, z: {z_hand}')
            # do inverse kinematics on arm position in realtime
            lens_real = self.inverse_kinematics(self.arm_position_resize[0] * x_hand, self.arm_position_resize[1] * y_hand, -z_hand)
            if lens_real is None:
                print('[Arm] ??? arm out of workspace')  # FIXME: actually possible
                return
            # apply algorithm on all chambers
            for index, chamber in enumerate(self.chambers):
                if index in range(0, 5):
                    len_predict = chamber.kalman(lens_real[index], lens_target[index], 0.1 * self.pressures[index] + 40 * chamber.lamda)
                else:
                    len_predict = chamber.kalman(lens_real[index], lens_target[index], chamber.lamda + 1.1249)
                print(f'    len_predict{index + 1}: {len_predict}, lambda{index + 1}: {chamber.lamda}')
                action_PID_now = chamber.GRLKPID()
                chamber.RLKMIC2(lens_target[index], len_predict, action_PID_now)


if __name__ == "__main__" and sys.argv[1] == 'auto':
    """run one of following command
      >>> python ./manipulate.py auto
      >>> python ./manipulate.py auto with_pwm
    """
    arm = Manipulator()
    arm.route_gen(20, 20)
    arm.RLKMIC()

if __name__ == '__main__' and sys.argv[1] == 'manual':
    """run one of following command
      >>> python ./manipulate.py manual
      >>> python ./manipulate.py manual with_pwm
    """
    arm = Manipulator()
    print('- input format: [x],[y],[z]\n- input q to quit')
    while True:
        command = input('input: ')
        if command == 'q':
            break
        elif command == '':
            continue
        if eval(f'arm.inverse_kinematics({command})'):
            arm.set_Pressures()
        else:
            print('not in workspace')
