import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wave
import pyaudio

#ideal file
folder = 'philharmonia/all-samples/clarinet/'
file = 'clarinet_Gs5_05_fortissimo_normal.wav'
sep = file[:-4].split('_')
spf = wave.open(folder+file, 'r')
signal = spf.readframes(-1)
signal = np.fromstring(signal, 'Int16')
intv = int(float(signal.shape[0])/8192)
new_shape = intv*8192
signal = signal[:new_shape].reshape(intv,-1)
#without noise
print intv
print np.abs(np.fft.rfft(signal,axis=1))
for i in range(4):
	FFT = np.abs(np.fft.rfft(signal,axis=1)[i])
	plt.plot(FFT)
	plt.show()



Format   = pyaudio.paInt16
Channels = 1
Rate     = spf.getframerate()
Chunk    = pow(2,10)

Rec_secs = signal.shape[0]/float(spf.getframerate())

#add noise
audio = pyaudio.PyAudio()
stream = audio.open(format=Format,
                    channels=Channels,
                    rate=Rate,
                    input=True,
                    frames_per_buffer=Chunk)
print 'recording...'
frames = []



for i in range(0, int(float(Rate)/Chunk*Rec_secs)):
	data = stream.read(Chunk)
	frames.append(data)
stream.stop_stream()
stream.close()
audio.terminate()
frames = b''.join(frames)
noise = np.fromstring(frames, 'Int16')


signal = signal+noise

#with noise
FFT = np.abs(np.fft.rfft(signal))
plt.plot(FFT)
plt.show()