#! /usr/bin/env python3

'''Soft manipulator driving module
provide pressure calculation functions with given end effector position
'''

from numpy import pi, sin, cos, arctan, exp, sign
from numpy.random import rand
import numpy as np
from scipy.optimize import root
from scipy.signal import cont2discrete
from pwm import PCA9685


class Manipulator():
    def __init__(self) -> None:
        self.d = 48.0  # FIXME: diameter of ?
        self.segBendLen = [0.0] * 3
        self.segElgLen = 0.0
        # the 10 channels are:
        # - upper bending segment chamber 1, 2, 3,
        # - lower bending segment chamber 1, 2, 3,
        # - elongation segment 1, 2, 3,
        # - hand
        self.pressures = [0.0] * 10
        self.water_pressure = 0.0
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
        # 7 Q matrices
        self.Q_Ms = np.zeros([7, 7, 7])
        # create 10 channel pwm module instance
        self.pwm = PCA9685()
        self.pwm.setPWMFreq()

    def inverse_kinematics(self, x: float, y: float, z: float) -> bool:
        """do inverse kinematics for the soft manipulator under the OBSS model
        with given position of end effector.

        If end point in the workspace, set length of each chamber of two bending
        segments and one elongation segment; if not, do nothing

        Parameters
        ----------
        x : float
            x of end effector  # FIXME: in what unit?
        y : float
            y of end effector
        z : float
            z of end effector, upward is the positive direction. initial z is -325

        Returns
        -------
        is_in_workspace: give bool value of whetcher the end point is in workspace
        """
        initBendLen = 107.0
        initElgLen = 102.0
        maxBendLen = 170.0
        maxElgLen = 200.0
        initZ = -325.0
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
                    self.segBendLen = [initBendLen] * 3
                    self.segElgLen = initElgLen + z
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
            [150, 0], args=(phiF, phiP, c, self.d, initBendLen), method='lm'
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
        if all(initBendLen < len < maxBendLen for len in segBendLen) and initElgLen < segElgLen < maxElgLen:
            # length of chamber 1, 2, 3 of two bending segments
            self.segBendLen = segBendLen
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
            self.pwm.setValue(channel, np.interp(p, [0, 500], [0, 1]) + 0.011)

    def set_Pressures(self):
        """set pressures of chambers `in the arm` (hand pressure not set here)
        then set pwm value of all channels
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
            self.segBendLen + [self.segElgLen]
        )
        # upper bending segment
        self.pressures[0:3] = map(lambda p: 0.98 * p - 0.6413, self.pressures[3:6])
        # elongation segment
        self.pressures[6:9] = [0.51789321 * self.segElgLen - 54.4050672] * 3
        # FIXME: hand pressure not set
        self.set_pwm()

    def kalman(self, A, B, C, D, Q, P, R, x_now, u, y_now, pp):
        Ad, Bd, Cd, Dd, _ = cont2discrete([A, B, C, D], 0.01)
        P = Ad * P * Ad.T + Bd * Q * Bd.T
        Mn = P * Cd.T / (Cd * P * Cd.T + R)
        P = (np.eye(2) - Mn * Cd) * P
        x_now = Ad * x_now + Bd * u
        x_next = x_now + Mn * (y_now - Cd * x_now - Dd)
        y_next = Cd * x_next + Dd
        p1 = D - y_now
        p2 = D - y_next
        p3 = np.ceil(max(pp, abs(p1)))
        if (
            p1 < -0.001 and 0 < p2 < 2 * p3
        ) or (
            p1 > 0.001 and -2 * p3 < p2 < 0
        ) or (
            np.sign(p1) == np.sign(p2) and abs(p1 - p2) > p3
        ):
            y_next = Cd * x_next + Dd - np.ceil(p1)
            sz = -D + y_next
        else:
            sz = Cd * P * Cd.T
        return x_next, y_next, sz, P, pp

    def GRLKPID(self, e, kp, R, Q, state, action, Q_m: np.ndarray):
        # 贪婪算法动作选择策略
        # FIXME?
        # if rand() >
        next_action = 0  # TODO: remove
        next_state = 0  # TODO:remove
        action_ranges = [-np.inf, -50, -1, -0.0001, 0.0001, 1, 50, np.inf]
        if True:
            for i in range(len(action_ranges) - 1):
                if action_ranges[i] <= e < action_ranges[i + 1]:
                    if i == 0:
                        next_action = 0
                    elif i == 6:
                        next_action = 6
                    elif i == 3:
                        next_action = np.argmax(Q_m[3, 2:5])
                    else:
                        next_action = np.argmax(Q_m[i, i - 1:i + 1])
                    next_state = i

        # 更新Q值表
        Q_m[state, action] = Q_m[state, action] + self.alpha * \
            (self.RA[next_state, next_action] + self.gama * Q_m[next_state, next_action] - Q_m[state, action])
        # 动作集
        action_dict = {
            0: (
                kp + (0.2 * exp(1 - 50 / abs(e)) + 0.02),
                R + (0.1 * exp(1 - 50 / abs(e)) + 0.01),
                Q - 0.1 * exp(1 - 50 / abs(e)) - 0.01
            ),
            1: (
                kp + (0.02 / exp(1 - 1 / 50) * exp(1 - 1 / abs(e)) + 0.002 * exp(0)),
                R + (0.01 / exp(1 - 1 / 50) * exp(1 - 1 / abs(e)) + 0.001),
                Q - (0.01 / exp(1 - 1 / 50) * exp(1 - 1 / abs(e)) + 0.001)
            ),
            2: (
                kp + (0.002 / exp(1 - 0.001) * exp(1 - 0.0001 / abs(e)) + 0.0002),
                R + (0.001 / exp(1 - 0.0001) * exp(1 - 0.0001 / abs(e)) + 0.0001),
                Q - (0.001 / exp(1 - 0.0001) * exp(1 - 0.0001 / abs(e)) + 0.0001)
            ),
            3: (
                kp - (0.0002 * exp(1 - 0.0001 / abs(e))) * sign(e),
                R - (0.0001 * exp(1 - 0.0001 / abs(e))) * sign(e),
                Q + (0.0001 * exp(1 - 0.0001 / abs(e))) * sign(e)
            ),
            4: (
                kp - (0.002 / exp(1 - 0.0001) * exp(1 - 0.0001 / abs(e)) + 0.0002),
                R - (0.001 / exp(1 - 0.0001) * exp(1 - 0.0001 / abs(e)) + 0.0001),
                Q + (0.001 / exp(1 - 0.0001) * exp(1 - 0.0001 / abs(e)) + 0.0001)
            ),
            5: (
                kp - (0.02 / exp(1 - 1 / 50) * exp(1 - 1 / abs(e)) + 0.0002),
                R - (0.01 / exp(1 - 1 / 50) + exp(1 - 1 / abs(e)) + 0.001),
                Q + (0.01 / exp(1 - 1 / 50) * exp(1 - 1 / abs(e)) + 0.001)
            ),
            6: (
                kp - 0.2 * exp(1 - 50 / abs(e)) - 0.02,
                R - 0.1 * exp(1 - 50 / abs(e)) - 0.01,
                Q + 0.1 * exp(1 - 50 / abs(e)) + 0.01
            )
        }
        kp, R, Q = action_dict[np.clip(next_action, 0, 6)]
        kp = np.clip(kp, 0.00001, np.inf)
        return kp, R, Q, next_state, next_action, action, Q_m

    def RLKMIC2(self):
        pass

    def RLKMIC(self):
        """RLK: reinforce learning kalman
        """
        pass


if __name__ == "__main__":
    arm = Manipulator()
    while True:
        x, y, z = input().strip().split(',')
        if eval(f'arm.inverse_kinematics({x},{y},{z})'):
            arm.set_Pressures()
        else:
            print('not in workspace')
