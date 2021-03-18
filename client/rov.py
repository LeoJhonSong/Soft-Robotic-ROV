import socket

# >>>>>> commands definition

# <<<<<< end commands definition

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
    """

    def __init__(self, state: str = 'initial'):
        self.state = state
        self.depth = 0
        self.gyro = Gyro()
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.bind(('127.0.0.1', ROV_SERVER_PORT))
        self.server_sock.listen(8)
        self.client_sock, _ = self.server_sock.accept()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.client_sock.close()
        self.server_sock.close()

    def send_command(self, command: bytes):
        """send command to ROV

        Parameters
        ----------
        command : bytes
            should be 27 bytes
        """
        self.client_sock.send(command)

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

    def print1(self):
        print('111')

    def print2(self):
        print('222')

    def printdefault(self):
        print('default')

    def switch(self):
        # TODO: update to key: str and each func has return state
        cases = {
            0: self.print1,
            1: self.print2,
            'default': self.printdefault
        }
        self.state = cases[self.state if (self.state in range(len(cases) - 1)) else 'default']()


if __name__ == '__main__':
    with Rov() as rov:
        rov.get()
