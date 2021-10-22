import sys
import os
import random
import struct
import numpy as np
read_file=open(sys.argv[1],'rb')
write_file=open(sys.argv[1]+"_test.dat", 'wb')
fftlength=131072
#生成单次batch数据
a=[]
i=0
while i<fftlength:
    signal_a=0
    signal_a+=np.cos(2 * np.pi * i / fftlength * 39000)
    signal_a+=np.cos(2 * np.pi * i / fftlength * 39001)
    signal_a+=np.cos(2 * np.pi * i / fftlength * 39002)
    signal_a+=np.sin(2 * np.pi * i / fftlength * 39003)
    signal_a+=np.cos(2 * np.pi * i / fftlength * 39004)
    signal_a+=np.sin(2 * np.pi * i / fftlength * 39005)
    signal_a+=np.cos(2 * np.pi * i / fftlength * 39006)
    signal_a+=np.sin(2 * np.pi * i / fftlength * 39007)
    signal_a+=np.sin(2 * np.pi * i / fftlength * 41008)
    signal_a+=np.sin(2 * np.pi * i / fftlength * 41009)
    signal_a+=np.sin(2 * np.pi * i / fftlength * 41012)
    signal_a+=np.sin(2 * np.pi * i / fftlength * 41013)
    signal_a=int((signal_a)*3)
    # 因为原始数据有两个通道，故这里需要写两次
    a.append(signal_a)
    a.append(signal_a)
    i+=1

b=[]
i=0
while i<fftlength:
    #signal_b=np.sin(2*np.pi*i/fftlength*55000)
    signal_b=int(0)
    # 因为原始数据有两个通道，故这里需要写两次
    b.append(signal_b)
    b.append(signal_b)
    i+=1

a=np.array(a,dtype='uint8')
b=np.array(b,dtype='uint8')
#写入多个batch
generated_data=np.array([],dtype='uint8')
total=9765
data_write=0
'''
while data_write<total-400:
    data_a=random.randint(30, 200)
    data_b=random.randint(30, 200)
    tmp = 0
    while tmp<data_a:
        original_data = np.array(struct.unpack("B" * (fftlength * 2), read_file.read(fftlength * 2)), dtype='uint8')
        outputdata=original_data+b
        write_file.write(struct.pack('B' * (fftlength*2), *outputdata))
        tmp+=1
    tmp = 0
    while tmp <data_b:
        original_data = np.array(struct.unpack("B" * (fftlength * 2), read_file.read(fftlength * 2)), dtype='uint8')
        outputdata = original_data+a
        write_file.write(struct.pack('B' * (fftlength * 2), *outputdata))
        tmp+=1
    data_write=data_write+data_a+data_b

tmp = 0
while tmp<total-data_write:
    original_data = np.array(struct.unpack("B" * (fftlength * 2), read_file.read(fftlength * 2)), dtype='uint8')
    outputdata=original_data+b
    write_file.write(struct.pack('B' * (fftlength*2), *outputdata))
    tmp+=1
'''
addnum = 100
while addnum<=400:
    tmp=0
    while tmp<addnum:
        #"B" * (fftlength * 2)是字符串运算，输出一个包含(fftlength * 2)个B的字符串
        original_data = np.array(struct.unpack("B" * (fftlength * 2), read_file.read(fftlength * 2)), dtype='uint8')
        outputdata=(original_data+a)%256
        #此处的乘号是starred expression，会被python迭代展开为多个参数
        write_file.write(struct.pack('B' * (fftlength*2), *outputdata))
        tmp+=1
    tmp = 0
    while tmp <20:
        original_data = np.array(struct.unpack("B" * (fftlength * 2), read_file.read(fftlength * 2)), dtype='uint8')
        outputdata = (original_data+b)%256
        write_file.write(struct.pack('B' * (fftlength * 2), *outputdata))
        tmp+=1
    data_write=data_write+addnum+20;
    addnum+=20
    print("data_write="+str(data_write))
tmp = 0
while tmp<total-data_write:
    original_data = np.array(struct.unpack("B" * (fftlength * 2), read_file.read(fftlength * 2)), dtype='uint8')
    outputdata=original_data+b
    write_file.write(struct.pack('B' * (fftlength*2), *outputdata))
    tmp+=1

read_file.close()
write_file.close()
