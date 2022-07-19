#open an actual data file and inject signals of some specific frequency
#only for data file of one channel
import argparse
import sys
import os
import random
import struct
import numpy as np
from matplotlib import pyplot

parser = argparse.ArgumentParser(description='Process some options.')
parser.add_argument('--input_type',type=str,help="数据的类型,与numpy所用的代号一致(即int8,>u2等,其中>u2表示大端无符号两字节整数)")
parser.add_argument('--file',type=str,help="输入文件")
parser.add_argument('--step',type=int,help="合并的样本点数")
parser.add_argument('--length',type=int,help="单张图片的横轴点数")
parser.add_argument('--pic_num',type=int,help="图片数")

args = parser.parse_args()

#input_type_size = struct.calcsize(args.data_type)
input_type_size=np.dtype(args.input_type).itemsize

filesize=os.path.getsize(args.file)
datafile = open(args.file,'rb')
print("file is large enough to draw "+str(filesize//(args.length*args.step*input_type_size))+" pictures")
if filesize<args.pic_num*args.length*args.step:
    print("input file is too small , exit")
    sys.exit(-1)

#用于创建文件名
name_formatter = '0'+str(len(str(args.pic_num)))+'d'

for pic_serial in range(args.pic_num):
    #假设原始数据为大端表示
    data=np.frombuffer(datafile.read(args.length*args.step*input_type_size),dtype=args.input_type)

    #求和
    data=data.reshape(args.length,args.step).sum(axis=1)

    #绘图
    #pyplot类似于opengl,是一个状态机,
    #pyplot.figure可以调整画布大小
    #pyplot.figure(dpi=300)
    #pyplot.plot在画布上画画
    #传入的data只需可以使用迭代器得到数字即可,因此numpy数组和python的list均可
    pyplot.plot(data)
    #pyplot.show()或pyplot.savefig(filename)将画布缓冲区的内容显示在屏幕上或保存为图片
    pyplot.savefig(args.file+format(pic_serial,name_formatter)+".png")
    #pyplot.clf清空画布的缓冲区
    pyplot.clf()

datafile.close()
