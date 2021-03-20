import socket

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


class Rov(object):
    """ROV master (run as a server)

    should be created with:
      >>> with Rov() as rov:
      >>>    pass

    Attributes
    ----------
    state : int
        current automation workflow state of ROV
    depth : float
        in meters
    info_word : bytes
        on/off word in form of: [6bytes] is_loop [7bytes] LED2 LED1
    joystick : bytes[4]
        Vx, Vy, direction, Vz
    """

    def __init__(self, state: str = 'initial'):
        self.state = state
        self.depth = 0
        self.is_landed = False
        self.info_word = bytes([0, 3])
        self.led_brightness = bytes([0, 0])
        self.joystick = bytes([0] * 4)
        self.gyro = Gyro()
        self.target = visual_info.Target()
        self.arm = visual_info.Arm()
        self.grasp_state = 'idle'
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.bind(('127.0.0.1', ROV_SERVER_PORT))
        self.server_sock.listen(8)
        self.client_sock, _ = self.server_sock.accept()

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

    def set_Vx(self, value: float):
        """set Vx and clear others

        Parameters
        ----------
        value : float
            in range [-1, 1]
        """
        self.velocity = bytes([int(127 + 127 * value), 0, 0, 0])

    def set_Vy(self, value: float):
        """set Vy and clear others

        Parameters
        ----------
        value : float
            in range [-1, 1]
        """
        self.velocity = bytes([0, int(127 + 127 * value), 0, 0])

    def set_Vz(self, value: float):
        """set Vz and clear others

        Parameters
        ----------
        value : float
            in range [-1, 1]
        """
        self.velocity = bytes([0, 0, 0, int(127 + 127 * value)])

    def set_direction(self, value: float):
        """set direction and clear others

        Parameters
        ----------
        value : float
            in range [-1, 1]
        """
        self.velocity = bytes([0, 0, int(127 + 127 * value), 0])

    def get(self):
        """receive the latest data from ROV

        a data package from ROV should be `bytes` of size 28, starts with 0xFE 0xFE,
        and ends with 1 byte XOR checksum and 0xFD OxFD
        """
        received = self.client_sock.recv(1024)[-RECV_DATA_SIZE:]
        self.depth = int.from_bytes(received, 'big') / 100
        gyro_list = [received[19:22].hex(), received[16:19].hex(), received[22:25].hex()]  # roll, pitch, yaw
        gyro_list = [(-1 if int(i[0]) else 1) * (int(i[1:4]) + int(i[4:]) / 100) for i in gyro_list]
        self.gyro.update(gyro_list)

    def land(self):
        """坐底
        """
        self.set_Vz(-1)
        # TODO: detect if is landed
        if self.is_landed:
            return 'grasp'
        else:
            return 'land'

    def grasp(self):
        """抓取
        """
        # if has target
        if self.arm.arm_is_working:
            self.grasp_state = 'ready'
        else:
            self.grasp_state = 'started'
        return 'grasp'
        # if time up & chance up
        # return 'cruise'
        # else
        self.grasp_state = 'idle'
        return 'cruise'

    def cruise(self):
        """巡航
        """
        # if time period
        return 'land'
        # else if detected
        return 'aim'

    def aim(self):
        """移动至目标处
        """
        # if target lost
        return 'land'
        # else if landed
        return 'grasp'

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
