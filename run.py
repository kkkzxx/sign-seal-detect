import socket
import sys

address = ('127.0.0.1', 10221)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(address)

s.send(sys.argv[1].encode())
while True:
    response_str = s.recv(1024).decode()
    if len(response_str) > 0 :
        print(response_str)
    if response_str == 'exit' :
        break
s.close()