#open an actual data file and inject signals of some specific frequency
#only for data file of one channel
import argparse
import sys
import os
import random
import struct
import numpy as np

parser = argparse.ArgumentParser(description='Process some options.')

parser.add_argument('--file', type=str, help="path of data file")
parser.add_argument('--freq', type=str, help="a list of frequency, such as f1,f2,f3,... ")
parser.add_argument('--fft_length', type=int, help="length of fft")
parser.add_argument('--time_interval', type=str, help="interval of added signal")
parser.add_argument('--input_type', type=str, help="type of signal,such as int8 , int16")
args = parser.parse_args()

freq=[int(num) for num in args.freq.split(',')]
time_interval=[int(num) for num in args.time_interval.split(',')]

read_file=open(args.file,'rb')
write_file=open(args.file+"_test.dat", 'wb')
print("input file is "+args.file)
print("output file is "+args.file+"_test.dat")

input_type_size=np.dtype(args.input_type).itemsize

#decide if file is large enouge
file_batch = os.path.getsize(args.file)//args.fft_length//input_type_size
if file_batch<sum(time_interval):
    print("input file is too small , exit")
    sys.exit(-1)

#generate virtual signal of one vatch
signal_single_fft=np.zeros([args.fft_length],dtype=args.input_type)

print("genetate virtual signal in one batch , "+str(len(freq))+" freqs , freqs are :")
print(freq)
for sample_point in range(args.fft_length):
    signal=0.0
    for freq_seq in range(len(freq)):
        signal+=np.cos(2.0 * np.pi * sample_point * freq[freq_seq]/ args.fft_length)
    
    signal=signal/len(freq)*(2**(input_type_size*8-4))
    signal_single_fft[sample_point]=signal;
#print("generated single fft data is:")
#print(signal_single_fft)

print("begin create new file")
for interval_serial in range(len(time_interval)):
    print("time interval is "+str(time_interval[interval_serial]))
    #if interval=0 ,this is not executed
    for batch_serial in range(time_interval[interval_serial]):
        output_data=np.frombuffer(read_file.read(args.fft_length*input_type_size),dtype=args.input_type)
        if(interval_serial%2==0):
            #add virtual signal to original signal
            output_data = output_data+signal_single_fft

        #write new data to file
        write_file.write(output_data.tobytes())

print("begin close file")
read_file.close()
write_file.close()
print("close file")
