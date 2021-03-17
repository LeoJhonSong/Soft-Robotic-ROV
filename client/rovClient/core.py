import socket

ROV_SERVER_PORT = 9090
RECV_DATA_SIZE = 28


class Rov(object):
    """ROV master (run as a server)

    should be created with:
      >>> with Rov() as rov:
      >>>    pass
    """

    def __init__(self, case=0):
        self.case = case
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.bind(('127.0.0.1', ROV_SERVER_PORT))
        self.server_sock.listen(8)
        self.client_sock, _ = self.server_sock.accept()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.client_sock.close()
        self.server_sock.close()

    def set(self, command: bytes):
        """send command to ROV

        Parameters
        ----------
        command : bytes
            should be 27 bytes
        """
        self.client_sock.send(command)

    def get(self):
        """receive the latest data from ROV

        a data package should be `bytes` of size 28
        """
        received = self.client_sock.recv(1024)[-RECV_DATA_SIZE:]

    def print1(self):
        print('111')

    def print2(self):
        print('222')

    def printdefault(self):
        print('default')

    def switch(self):
        cases = {
            0: self.print1,
            1: self.print2,
            'default': self.printdefault
        }
        self.case = cases[self.case if (self.case in range(len(cases) - 1)) else 'default']()


if __name__ == '__main__':
    with Rov() as rov:
        rov.get()
