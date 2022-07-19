import struct
import sys
import os
#import pdb
import numpy as np
import argparse
from matplotlib import pyplot
parser = argparse.ArgumentParser(description='Process some options.')

parser.add_argument('--channel_num',type=int, help="文件包含的通道数量")
parser.add_argument('--file',type=str,help="文件路径")
parser.add_argument('--fre_add',type=int,help="频率加和数")
parser.add_argument('--time_add',type=int,help="时间加和数")

args = parser.parse_args()

remain_batch=os.path.getsize(args.file)//(args.channel_num//8)

#规定每次读取的batch数，用来控制程序的内存使用量
per_batch=1024

f = open(args.file, 'rb')

#创建一个包含channel_num//fre_add个子数组的空数组
data=np.array([])
for i in range(args.channel_num//args.fre_add-1):
    #保持原有数组的结构，“纵向堆叠起来”(相当于把这些数组作为元素放到一个新数组里)
    data=np.vstack((data,[]))

#pdb.set_trace()
while(remain_batch>=per_batch):
    print("read data to buffer")
    data_tmp=np.frombuffer(f.read(args.channel_num*per_batch//8),dtype='uint8')
    data_tmp=np.unpackbits(data_tmp)
    #按照行优先的内存排布方式重新组织数组
    data_tmp=data_tmp.reshape(per_batch,args.channel_num//args.fre_add,args.fre_add)
    #在频率方向加和
    data_tmp=data_tmp.sum(axis=2)
    data_tmp=data_tmp.T
    # 在时间上也加和一下
    data_tmp=data_tmp.reshape(args.channel_num//args.fre_add,per_batch//args.time_add,args.time_add)
    data_tmp=data_tmp.sum(axis=2)
    #把对这些batch的求和堆叠过来
    data=np.hstack((data,data_tmp))
    #计算剩余的batch数
    remain_batch=remain_batch-per_batch
#有余数
if(remain_batch//args.time_add>0):
    print("read remain data to buffer")
    #确保remain_batch被step整除
    remain_batch = (remain_batch//args.time_add)*args.time_add
    data_tmp = np.frombuffer(f.read(args.channel_num*remain_batch//8),dtype='uint8')
    data_tmp = np.unpackbits(data_tmp)
    data_tmp = data_tmp.reshape(remain_batch, args.channel_num//args.fre_add, args.fre_add)
    data_tmp = data_tmp.sum(axis=2)
    data_tmp = data_tmp.T
    # 在时间上也加和一下
    data_tmp = data_tmp.reshape(args.channel_num//args.fre_add, remain_batch//args.time_add, args.time_add)
    data_tmp = data_tmp.sum(axis=2)
    data = np.hstack((data, data_tmp))

pyplot.figure(dpi=300)
pyplot.imshow(data)
pyplot.savefig(args.file+".png")
f.close()

