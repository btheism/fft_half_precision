#读取一个实际的数据文件,在特定频率段注入周期性频率信号
#生成的是单通道数据
import argparse
import sys
import os
import random
import struct
import numpy as np

parser = argparse.ArgumentParser(description='Process some options.')

parser.add_argument('--file', type=str, help="输入文件路径")
parser.add_argument('--freq', type=str, help="注入的信号的频率列表,语法为f1,f2,f3,... ,不得有空格")
parser.add_argument('--fft_length', type=int, help="fft变换的长度")
parser.add_argument('--time_interval', type=str, help="注入信号的间隔,以fft的长度为单位,语法与freq相同,奇数位表示有信号,偶数位表示无信号")
parser.add_argument('--input_type', type=str, help="信号的类型,b表示8位有符号数,h表示16位有符号数,大写表示无符号数")
args = parser.parse_args()

freq=[int(num) for num in args.freq.split(',')]
time_interval=[int(num) for num in args.time_interval.split(',')]

read_file=open(args.file,'rb')
write_file=open(args.file+"_test.dat", 'wb')

input_type_size=np.dtype(args.input_type).itemsize

#判断文件长度是否够用
file_batch = os.path.getsize(args.file)//args.fft_length//input_type_size
if file_batch<sum(time_interval):
    print("input file is too small , exit")
    sys.exit(-1)

#生成单个fft_length长度的数据
signal_single_fft=np.zeros([args.fft_length],dtype=args.input_type)

for sample_point in range(args.fft_length):
    signal=0.0
    for freq_seq in range(len(freq)):
        signal+=np.cos(2.0 * np.pi * sample_point * freq[freq_seq]/ args.fft_length)
    
    signal=signal/len(freq)*(2**(input_type_size*8-2))
    signal_single_fft[sample_point]+=signal;
#print("generated single fft data is:")
#print(signal_single_fft)

for interval_serial in range(len(time_interval)):
    #如果interval是0的话会直接跳过
    for batch_serial in range(time_interval[interval_serial]):
        output_data=np.frombuffer(read_file.read(args.fft_length*input_type_size),dtype=args.input_type)
        if(interval_serial%2==0):
            #把生成的信号加在原始数据上
            output_data = output_data+signal_single_fft

        #写入文件
        write_file.write(output_data.tobytes())

read_file.close()
write_file.close()
