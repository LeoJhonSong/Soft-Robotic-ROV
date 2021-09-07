#! /usr/bin/env python3
"""roughly test delay between server and client
1. on server side:
>>> delay_test.py server
2. on client side:
>>> delay_test.py client x.x.x.x  # ip of server
"""

import sys
import time
import socket

PORT = 9000
count = 100

if sys.argv[1] == 'client':
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((sys.argv[2], PORT))
        sum = 0
        for i in range(count):
            s.send(bytes(str(time.time()).encode()))
            sum += (time.time() - float(s.recv(1024).decode("utf-8"))) * 1000
        s.close()
        print(f'average delay of {count} tries: {sum / count:.03f}ms')

if sys.argv[1] == 'server':
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('0.0.0.0', PORT))
        s.listen()
        connect, client_addr = s.accept()
        for i in range(count):
            connect.send(connect.recv(1024))
        connect.close()
