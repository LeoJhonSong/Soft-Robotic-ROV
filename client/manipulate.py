#! /usr/bin/env python3

'''Soft manipulator driving module
provide pressure calculation functions with given end effector position
'''

from math import atan, cos, pi, sin
import numpy as np
from scipy.optimize import root
# from pwm import PCA9685


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
        # create 10 channel pwm module instance
        # self.pwm = PCA9685()
        # self.pwm.setPWMFreq()

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
        initBendLen = 109.0
        initElgLen = 102.0
        maxBendLen = 170.0
        maxElgLen = 200.0
        initZ = -325.0
        # determine phi based on x, y
        if x < 0:
            phi = -atan(y / x) + pi
        elif x > 0:
            phi = -atan(y / x)
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


if __name__ == "__main__":
    arm = Manipulator()
    while True:
        x, y, z = input().strip().split(',')
        if eval(f'arm.inverse_kinematics({x},{y},{z})'):
            arm.set_Pressures()
        else:
            print('not in workspace')
