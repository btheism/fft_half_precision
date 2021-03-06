#该程序为其他人提供的示例程序，存档在此
from sigpyproc.Readers import FilReader as filterbank
import numpy as np
import sys, os
from pylab import *

fil = filterbank(sys.argv[1])

fch1 = fil.header['fch1']
df = fil.header['foff']
fmin = fil.header['fbottom']
fmax = fil.header['ftop']
nsamp = fil.header['nsamples']
tsamp = fil.header['tsamp']
nf = fil.header['nchans']
tstart = fil.header['tstart']
nchans = fil.header['nchans']
hdrlen = fil.header['hdrlen']

print('fch1:', fch1)
print('df:', df)
print('fmin:', fmin)
print('fmax:', fmax)
print('nsamp:', nsamp)
print('tsamp:', tsamp)
print('nf:', nf)
print('nchans:', nchans)
print('tstart:', tstart)

print('number of channels:', nf)
fil._file.seek(hdrlen)

nsamp=(nsamp//256)*256
data = fil._file.cread(nchans*nsamp)
data = np.array(data.reshape((nsamp, nf)).transpose(), order='C')
print('average value in data:', data.mean(), data.std(), data.max())


l, m = data.shape
#data = data.reshape( (l, m/128, 128) ).sum(axis=2)
data = data.reshape( (l//4, 4, m) ).sum(axis=1)
data = data.reshape( (l//4, m//64, 64) ).sum(axis=2)
print(data.shape)

idx = np.ones(l//4)
#idx[280:300] = 0
#idx[560:600] = 0

#imshow(data[2800:,25:], aspect='auto', origin='bottomleft')
#imshow(data[:,20000:30000], aspect='auto')#, origin='bottomleft')
#imshow(data[:,:], aspect='auto', extent=[0, m*tsamp, 1000., 1500])#, origin='bottomleft')
#imshow((data[:,100:].T*idx[:].T).T, aspect='auto', extent=[100*128*tsamp, m*tsamp, 1000., 1500], cmap='gray')

#imshow((data[:,:].T*idx[:].T).T, aspect='auto', extent=[0*128*tsamp, m*tsamp, 1000., 1500], cmap='gray')
imshow((data[:,:].T*idx[:].T).T, aspect='auto', extent=[0*64*tsamp, m*tsamp, 1000., 1500])
xlabel('time (s)')
ylabel('frequency (MHz)')
show()
