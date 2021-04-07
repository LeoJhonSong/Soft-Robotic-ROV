import socket
import time

import visual_info

ROV_SERVER_PORT = 9090
RECV_DATA_SIZE = 28


class Gyro(object):
    """Gyroscope sensor class

    Attributes
    ----------
    x : float
        roll (横滚角)
    y : float
        pitch (俯仰角)
    z : float
        yaw (偏航角)
    """

    def __init__(self) -> None:
        self.x = 0
        self.y = 0
        self.z = 0

    def update(self, gyro_list: list) -> None:
        self.x = gyro_list[0]
        self.y = gyro_list[1]
        self.z = gyro_list[2]


class Depth_sensor(object):
    """depth sensor for ROV

    Attributes
    ----------
    depth : float
        in meters
    """

    def __init__(self):
        self.depth = 0.0
        self.old_depth = 0.0
        self.count = 0
        self.count_thresh = 10
        self.diff_thresh = 0.03  # 3cm
        self.is_landed = False

    def land_check(self) -> None:
        if abs(self.old_depth - self.depth) < self.diff_thresh:
            self.count += 1
            if self.count > self.count_thresh:
                print('[ROV] landed!')
                self.is_landed = True
            else:
                self.is_landed = False
        else:  # 当深度变化幅度超过阈值, 判定未坐底并归零稳定计次
            self.count = 0
            self.is_landed = False


class Rov(object):
    """ROV master (run as a server)

    should be created with:
      >>> with Rov() as rov:
      >>>    pass

    Attributes
    ----------
    state : int
        current automation workflow state of ROV
    info_word : bytes
        on/off word in form of: [6bytes] is_loop [7bytes] LED2 LED1
    joystick : bytes[4]
        Vx, Vy, direction, Vz
    """

    def __init__(self, state: str = 'initial'):
        self.state = state
        self.depth_sensor = Depth_sensor()
        self.info_word = bytes([0, 3])
        self.led_brightness = bytes([0, 0])
        self.joystick = bytes([0] * 4)
        self.gyro = Gyro()
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
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.bind(('127.0.0.1', ROV_SERVER_PORT))
        self.server_sock.listen(8)
        print('[ROV] connecting ROV...')
        self.client_sock, _ = self.server_sock.accept()
        print('[ROV] connected')
        self.set_led(950)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.client_sock.close()
        self.server_sock.close()

    def send_command(self):
        """send command to ROV

        """
        temp = b'\xFE\xFE' + \
            self.info_word + \
            self.led_brightness + \
            bytes([0] * 12) + \
            self.joystick
        checksum = 0
        for i in temp:
            checksum ^= i
        self.client_sock.send(temp + bytes([checksum]) + b'\xFD\xFD')

    def set_control_mode(self, is_loop: bool):
        """set rov motor speed control to close loop PID control or open loop control

        Parameters
        ----------
        is_loop : bool
            True for close loop control, False for open loop
        """
        self.info_word = bytes([is_loop * 2, 3])

    def set_led(self, value: int):
        """set brightness of all four LEDs

        Parameters
        ----------
        value : int
            min is 0, max is 950
        """
        self.led_brightness = bytes.fromhex(f'{value:04x}')
        self.send_command()
        print(f'[ROV] led brightness set to {int(value / 9.5)}%')

    def set_Vx(self, value: float):
        """set Vx and clear others

        Parameters
        ----------
        value : float
            in range (backward)[-1, 1](forward)
        """
        self.joystick = bytes([int(127 + 127 * value), 0, 0, 0])
        self.send_command()
        if value:
            print(f'[ROV] going {"forward" if value > 0 else "backward"} with {int(abs(value) * 100)}% speed')
        else:
            print('[ROV] stopped')

    def set_Vy(self, value: float):
        """set Vy and clear others

        Parameters
        ----------
        value : float
            in range (right)[-1, 1](left)
        """
        self.joystick = bytes([0, int(127 + 127 * value), 0, 0])
        self.send_command()
        if value:
            print(f'[ROV] going {"left" if value > 0 else "right"} with {int(abs(value) * 100)}% speed')
        else:
            print('[ROV] stopped')

    def set_Vz(self, value: float):
        """set Vz and clear others

        Parameters
        ----------
        value : float
            in range (down)[-1, 1](up)
        """
        self.joystick = bytes([0, 0, 0, int(127 + 127 * value)])
        self.send_command()
        if value:
            print(f'[ROV] going {"up" if value > 0 else "down"} with {int(abs(value) * 100)}% speed')
        else:
            print('[ROV] stopped')

    def set_direction(self, value: float):
        """set direction and clear others

        Parameters
        ----------
        value : float
            in range (right)[-1, 1](left)
        """
        self.joystick = bytes([0, 0, int(127 + 127 * value), 0])
        self.send_command()
        if value:
            print(f'[ROV] turning {"left" if value > 0 else "right"} with {int(abs(value) * 100)}% speed')
        else:
            print('[ROV] stopped')

    def set_move(self, velocity: list):
        """set multi-direction values

        Parameters
        ----------
        velocity : list
            a list of [Vx, Vy, direction, Vz] in range [-1, 1]
        """
        self.joystick = bytes([int(127 + 127 * item) for item in velocity])
        self.send_command()

    def get(self):
        """receive the latest data from ROV

        a data package from ROV should be `bytes` of size 28, starts with 0xFE 0xFE,
        and ends with 1 byte XOR checksum and 0xFD OxFD
        """
        received = self.client_sock.recv(1024)[-RECV_DATA_SIZE:]
        self.depth_sensor.depth = int.from_bytes(received, 'big') / 100  # in meters
        gyro_list = [received[19:22].hex(), received[16:19].hex(), received[22:25].hex()]  # roll, pitch, yaw
        gyro_list = [(-1 if int(i[0]) else 1) * (int(i[1:4]) + int(i[4:]) / 100) for i in gyro_list]
        self.gyro.update(gyro_list)

    def land(self):
        """坐底
        """
        self.set_Vz(-1)
        if self.depth_sensor.is_landed:
            return 'grasp'
        else:
            return 'land'

    def grasp(self):
        """抓取
        """
        if self.target.has_target:
            if not self.arm.arm_is_working:
                self.grasp_state = 'ready'
                return 'grasp'
            else:
                if self.arm.chances:
                    if time.time() - self.arm.start_time > self.arm.time_limit:
                        self.arm.chances[0] -= 1
                    self.grasp_state = 'activated'
                    return 'grasp'
        # has no target / target lost / no more chances
        self.grasp_state = 'idle'
        self.arm.chances[0] = self.arm.chances[1]  # reset chances
        # 往前荡一下, 确保目标进袋
        self.set_Vx(1)
        time.sleep(1)
        # float up to cruise hight
        self.set_Vz(1)
        time.sleep(0.5)
        return 'cruise'

    def cruise(self):
        """巡航
        """
        if self.target.has_target:
            self.cruise_time = 0
            # 发现目标后一个后撤步!
            self.set_move([-0.7, 0, 0, -0.99])
            time.sleep(1)  # 1s
            return 'aim'
        # if reach time limit
        elif time.time() - self.cruise_time > list(self.cruise_path.keys())[-1] * self.cruise_periods:
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

    def aim(self):
        """瞄准, 移动至目标处
        """
        offset_x = 0.5
        offset_y = 0.7
        # if target lost
        if not self.target.has_target:
            return 'land'
        else:
            # 已坐底
            if self.depth_sensor.is_landed:
                return 'grasp'
            else:
                # 最多4次机会
                for i in range(4):
                    while True:
                        # FIXME: a sleep may needed to limit the fps (or not)
                        pass
                # 放弃瞄准, 转坐底
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
    with Rov() as rov:
        rov.get()
