import socket
import time
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# 绑定端口:
addr=('127.0.0.1' , 9999)
s.bind(addr)
print('listen UDP on 9999...') 
while True:
    # 接收数据:
    data, addr = s.recvfrom(1024);
    #print(data[8:])
    print(data)
