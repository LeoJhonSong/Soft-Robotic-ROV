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

if sys.argv[1] == 'client':
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((sys.argv[2], 8000))
        s.send(bytes(str(time.time()).encode()))
        print(f'{(time.time() - float(s.recv(1024).decode("utf-8"))) * 1000:.03f}ms')

if sys.argv[1] == 'server':
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('localhost', 8000))
        s.listen()
        connect, client_addr = s.accept()
        connect.send(connect.recv(1024))
