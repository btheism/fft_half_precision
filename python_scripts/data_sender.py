import socket
import time
import struct
import sys
import os

per_length=4096
#per_length=20
sleep_time=0.000006
#sleep_time=0.5

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
f_size=os.path.getsize(sys.argv[1])
f_stream = open(sys.argv[1],'rb')
# 绑定端口:
addr=('127.0.0.1' , 9999)
packet_serial_number=0;
while True:
    # 发送数据:
    if(f_size<per_length):
        f_stream.close()
        f_stream = open(sys.argv[1],'rb')
        f_size=os.path.getsize(sys.argv[1])
    
    data=struct.pack('Q',packet_serial_number)+f_stream.read(per_length)
    #data=struct.pack('Q',packet_serial_number)
    f_size-=per_length
    time.sleep(sleep_time)
    
    print('send %d packet to %s:%s.' % (packet_serial_number ,addr[0],addr[1]))
    s.sendto(data, addr)
    packet_serial_number+=1;
 

#f.close()
