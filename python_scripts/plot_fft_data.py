#旧文件备份，无实际用途
import struct
import sys
import os
import numpy as np
from matplotlib import pyplot
fftlength=32768
batch_total=os.path.getsize(sys.argv[1])//fftlength//4
batch=1024
fre_add=int(sys.argv[2])
time_add=int(sys.argv[3])
#batch=os.path.getsize(sys.argv[1])//fftlength//2
f = open(sys.argv[1], 'rb')
data=np.array([])
for i in range(int(fftlength/fre_add)-1):
    data=np.vstack((data,[]))
while(batch_total>=batch):
    data_tmp=np.array(struct.unpack("f"*(fftlength*batch),f.read(fftlength*batch*4)))
    #data=np.array(struct.unpack("e"*(fftlength*batch),f.read(fftlength*batch*2)))
    data_tmp=data_tmp.reshape(batch,int(fftlength/fre_add),fre_add)
    data_tmp=data_tmp.sum(axis=2)
    data_tmp=data_tmp.T
    data_tmp=data_tmp.reshape(int(fftlength/fre_add),int(batch/time_add),time_add)
    data_tmp=data_tmp.sum(axis=2)
    data=np.hstack((data,data_tmp))
    batch_total=batch_total-batch
if(int(batch_total/time_add)>0):
    batch=int(batch_total/time_add)*time_add
    data_tmp=np.array(struct.unpack("f"*(fftlength*batch),f.read(fftlength*batch*4)))
    #data=np.array(struct.unpack("e"*(fftlength*batch),f.read(fftlength*batch*2)))
    data_tmp=data_tmp.reshape(batch,int(fftlength/ fre_add), fre_add)
    data_tmp=data_tmp.sum(axis=2)
    data_tmp=data_tmp.T
    data_tmp=data_tmp.reshape(int(fftlength/fre_add),int(batch/time_add),time_add)
    data_tmp=data_tmp.sum(axis=2)
    data=np.hstack((data,data_tmp))

#data=data/1e10
pyplot.figure(figsize=(64,48))
#pyplot.imshow(data.T)
pyplot.imshow(data)
pyplot.savefig(sys.argv[1]+".png")
f.close()
