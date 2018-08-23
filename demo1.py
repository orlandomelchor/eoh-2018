#! /usr/bin/env python
######################
#Author: Orlando Melchor-Alonso
#Last Updated: 2018/07/23
######################
#Objective: plot the signal and fft of a sound file or a short segment from the mic
######################

import pyaudio
import wave
import numpy as np
import matplotlib.pyplot as plt
import sys
import time

##########read the signal using the mic##########
Format   = pyaudio.paInt16
Channels = 1                #stero currently not supported

#parameters below here must be powers of 2 for a faster fft
Rate     = pow(2,16)        #near double the highest audible frequency
Chunk    = pow(2,10)        #1024 bit
Rec_secs = 2	            #duration in seconds (more secs means more samples so must reshape using this)

#parameters below here do not have to be powers of 2
min_freq = 29              #min detected frequency
max_freq = 20000 	        #max detected frequency
cents    = 200		        #cents around the peak frequency
thresh   = .05              #percentage from peak frequency as threshold
numHarm  = int(sys.argv[1]) #number of harmonics to obtain

#detects the peak frequencies in an FFT
def harmonic_detector(fft, numHarm, rec=Rec_secs):
	harmonics = np.zeros(numHarm)
	FFT = np.copy(fft)
	for i in range(numHarm):
		#get the current peak
		curr_max = np.argmax(FFT)
		low_idx  = int(pow(2,-cents/1200.)*curr_max)
		high_idx = int(pow(2,cents/1200.)*curr_max)
		#exclude in and around current peak in future searches
		FFT[low_idx:high_idx] = 0
		harmonics[i] = curr_max
	#must resize harmonics because there are more samples with more Rec_secs
	return harmonics / rec

#filters the FFT using from threshold and low/high pass filter
def filter(FFT):
	#must first resize filters 
	low_pass  = int(min_freq*Rec_secs)
	high_pass = int(max_freq*Rec_secs)
	#low pass filter
	FFT[:low_pass] = 0
	#high paas filter
	FFT[high_pass:] = 0
	print high_pass
	print FFT.shape[0]
	#remove noise within a certain percent of max
	#FFT[FFT < np.max(FFT)*thresh] = 0

def from_mic():
	#start stream
	audio = pyaudio.PyAudio()
	stream = audio.open(format=Format,
	                    channels=Channels,
	                    rate=Rate,
	                    input=True,
	                    frames_per_buffer=Chunk)
	print 'recording...'

	#get the signal from the mic
	frames = []
	for i in range(0, int(Rate/Chunk*Rec_secs)):
		data = stream.read(Chunk)
		frames.append(data)
	stream.stop_stream()
	stream.close()
	audio.terminate()
	frames = b''.join(frames)
	signal = np.fromstring(frames, 'Int16')

	#plot intial signal over time
	plt.figure(1)
	plt.title('original signal')
	plt.plot(signal)
	plt.show()

	#use real one dimensional fft and filter
	FFT = np.abs(np.fft.rfft(signal))
	filter(FFT)

	#get the frequencies
	freqs = np.arange(FFT.shape[0])/Rec_secs

	#determine harmonics
	harmonics = harmonic_detector(FFT, numHarm)
	print(harmonics)

	#plot the fft after filtering
	plt.figure(2)
	plt.title('fft of signal wave')
	plt.plot(freqs, FFT)
	plt.show()

##########read the signal from a .wav file##########
def from_file(file):
	spf = wave.open(file, 'r')
	Rate = spf.getframerate()
	signal = spf.readframes(-1)
	signal = np.fromstring(signal, 'Int16')

	#fft works faster for powers of 2 so pad the end with zeros
	padding = pow(2,int(np.log2(signal.shape[0]))+1)-signal.shape[0]
	signal = np.pad(signal, (0,padding), 'constant')

	Rec_secs = signal.shape[0]/float(Rate)

	#plot intial signal over time
	plt.figure(3)
	plt.title('original signal')
	plt.plot(signal)
	plt.show()


	#use real one dimensional fft and filter
	FFT = np.abs(np.fft.rfft(signal))
	filter(FFT)

	#get the frequencies
	freqs = np.arange(FFT.shape[0])/Rec_secs


	#determine harmonics
	harmonics = harmonic_detector(FFT, numHarm, Rec_secs)
	print(harmonics)

	#plot the fft after filtering
	plt.figure(4)
	plt.title('fft of signal wave')
	plt.plot(freqs, FFT)
	plt.show()

if len(sys.argv) == 3:
	from_file(sys.argv[2])
else:
	from_mic()