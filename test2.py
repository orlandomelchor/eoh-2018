import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time as t

def freq_to_MIDI(f):
	if f==0:
		return 0
	return 69 + 12*np.log2(f/440.0)


def MIDI_to_freq(n):
	return 440.0*pow(2,(n-69)/12.)

def top_window(n,cents):
	f = 440.0*np.power(2,(n-69)/12.)
	high = (pow(2,cents/1200.)*f)*time
	return high.astype(int)

def window(n,cents):
	f = 440.0*np.power(2,(n-69)/12.)
	high = ((pow(2,cents/1200.)*f)*time).reshape(-1,1)
	low = ((pow(2,-cents/1200.)*f)*time).reshape(-1,1)
	return np.hstack((low.astype(int), high.astype(int)))




header = pd.read_csv('philharmonia_signals.csv',nrows=1,header=None).values
time = header[0,-1]
# Hann window function ##https://en.wikipedia.org/wiki/Hann_function
signals = pd.read_csv('philharmonia_signals.csv',nrows=10000,usecols=np.arange(7,header.shape[1])).values[:,1:]
Hann= 0.5* (1-np.cos(np.linspace(0,2*np.pi, signals.shape[1],False)))	#end point not included
signals = signals*Hann

FFT = np.abs(np.fft.rfft(signals,axis=1))

frequencies = np.arange(FFT.shape[1])/time
max_MIDI = 144
cents = 50
midi_frequencies = np.arange(max_MIDI)
top_windows = top_window(midi_frequencies,cents)
t0 = t.time()
adj_FFT = np.array([np.max(e,axis=1) if e.shape[1]>1 else np.zeros(e.shape[0]) for e in np.split(FFT,top_windows,axis=1)[:-1]]).T
print t.time() - t0
'''
windows = window(midi_frequencies,cents)
t0 = t.time()
c = np.r_['-1', np.zeros(FFT.shape[0]).reshape(-1,1), np.cumsum(FFT,axis=1)][:,windows]
midi_FFT = c[:,:,1] - c[:,:,0]
print t.time() - t0
'''
n_frequencies = np.array([freq_to_MIDI(f) for f in frequencies])




data_frame = pd.DataFrame(adj_FFT)
data_frame.to_csv('reduced_resolution_FFT.csv')

print FFT.shape
print adj_FFT.shape
for i in range(FFT.shape[0]):
	plt.subplot(1, 4, 1)
	plt.title('Original signal')
	plt.xlabel('time')
	plt.ylabel('Amplitude')
	plt.plot(np.arange(signals.shape[1]), signals[i]*Hann)
	plt.subplot(1, 4, 2)
	plt.title('FFT in Hertz')
	plt.xlabel('Frequency in Hertz')
	plt.ylabel('Amplitude')
	plt.plot(frequencies, FFT[i])
	plt.subplot(1, 4, 3)
	plt.title('FFT in Midi {}'.format(FFT.shape[1]))
	plt.xlabel('Frequency in Midi with no information loss')
	plt.ylabel('Amplitude')
	plt.plot(n_frequencies, FFT[i])
	plt.subplot(1, 4, 4)
	plt.title('FFT in Midi with lower resolution {}'.format(adj_FFT.shape[1]))
	plt.xlabel('Frequency in Midi')
	plt.ylabel('Amplitude')
	plt.plot(midi_frequencies, adj_FFT[i])
	plt.show()

