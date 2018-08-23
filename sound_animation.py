#! /usr/bin/env python
######################
#Author: Sarah Habib
#		 Orlando Melchor-Alonso
#Last Updated: 2018/07/24
######################
#Objective: create an animation for the components and superposition of sound signals
######################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import pi, sin
import soundtest as st
import sys
import time
import pyaudio
from sklearn.ensemble import RandomForestClassifier
import pickle

#directories
model_dir = 'models/'

#number of harmonics to read
numNotes = int(sys.argv[1])

##########FFT of mic input##########
Format   = pyaudio.paInt16
Channels = 1                  #stero currently not supported

#parameters below here must be powers of 2 for a faster fft
Rate       = pow(2,16)        #near double the highest audible frequency
Chunk      = pow(2,10)        #1024 bit
Rec_secs   = pow(2,-1)        #duration in seconds (more secs means more samples so must reshape using this)
noise_secs = pow(2,3)		  #duration in seconds for nosie gathering (more secs means more samples so must reshape using this)

#parameters below here do not have to be powers of 2
min_freq = 29                #min detected frequency (lowest audible)
max_freq = 20000 	          #max detected frequency (highest audible)
cents    = 200		          #amount to remove around peak frequency (in cents) once it is found

noteNames='C C# D D# E F F# G G# A A# B'.split() # names for all the notes ## are there more?

def freq_to_MIDI(f): 
	if f == 0:
		return 0
	'''Converts frequency to MIDI number'''
	return 69 + 12*np.log2(f/440.0)

def MIDI_to_name(n): 
	if n == 0:
		return ''
	'''Converts MIDI number to note names'''
	return noteNames[int(n % 12)] + str(int(n/12) - 1)


#collect signal every Rec_secs seconds
def collect_signal(rec=Rec_secs):
	frames = []
	for i in range(0, int(Rate/Chunk*rec)):
		data = stream.read(Chunk)
		frames.append(data)
	frames = b''.join(frames)
	signal = np.fromstring(frames, 'Int16')
	return signal

#detect the first n harmonics from mic
def sample_sound(n):
	#filter background noise by subtracting it out
	signal = collect_signal()
	#use real one dimensional fft
	FFT = np.abs(np.fft.rfft(signal,norm='ortho'))
	#filter above certain threshold and remove low and high pass
	filter_FFT(FFT)
	#determine harmonics
	f0, amp = harmonic_detector(FFT, n)
	#get all harmonics above the root
	f0_idx = model.predict([f0])
	f = f0[f0>=f0[f0_idx]]
	amp = amp[f0>=f0[f0_idx]]
	#pad the missing harmonics with zeros
	padding = f0.shape[0] - f.shape[0]
	f = np.pad(f, (0,padding), 'constant')
	amp = np.pad(amp, (0,padding), 'constant')
	return f, amp

#gather noise to remove from the signal
def collect_noise(rec=noise_secs):
	noise = collect_signal(rec)
	rec_mult = noise.shape[0]/signal.shape[0]
	#take the average of the noise signal over equal Rec_secs intervals
	noise = noise.reshape(rec_mult,-1)
	return np.mean(np.abs(np.fft.rfft(noise)),axis=0)

#filters the FFT using from threshold and low/high pass filter
def filter_FFT(FFT):
	#must first resize filters 
	low_pass  = int(min_freq*Rec_secs)
	high_pass = int(max_freq*Rec_secs)
	#low pass filter
	FFT[:low_pass] = 0
	#high paas filter
	FFT[high_pass:] = 0
	#remove noise within a certain percent of max
	FFT[FFT < thresh] = 0

#detects the peak frequencies in an FFT
def harmonic_detector(FFT, numHarm):
	harmonics = np.zeros(numHarm)
	amplitude = np.zeros(numHarm)
	for i in range(numHarm):
		#get the current peak
		curr_max = np.argmax(FFT)
		low_idx  = int(pow(2,-cents/1200.)*curr_max)
		high_idx = int(pow(2,cents/1200.)*curr_max)
		#exclude in and around current peak in future searches
		amplitude[i] = FFT[curr_max]
		FFT[low_idx:high_idx] = 0
		#the harmonic is too high to be detected never too low
		harmonics[i] = curr_max / Rec_secs
	#must resize harmonics because there are more samples with more Rec_secs
	return harmonics, amplitude

#random forest model for fundamental detection
print 'loading fundamental frequency model...'
filename = '{}fundamental_frequency_{}.sav'.format(model_dir,numNotes)
model = pickle.load(open(filename, 'rb'))

print 'initializing signal detector...'
#start stream
audio = pyaudio.PyAudio()
stream = audio.open(format=Format,
                    channels=Channels,
                    rate=Rate,
                    input=True,
                    frames_per_buffer=Chunk)
#gather initial signal for initialization
signal = collect_signal()

#gathering background noise
print 'do not play collecting background noise...'
noise = collect_noise()

#maximum threshold is the peak of the environment
thresh = 0

xdata = np.linspace(0, 2*np.pi, 30000)
j = np.size(xdata)

xdata = np.reshape(xdata, (1, j))
fig = plt.figure()
ax = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

def update(frame):
	ax.clear()
	ax2.clear()
	#A Major Chord exhibit (A, C#, E)
	#f, name = np.asarray([440.0, 554.37, 659.25]), ["A","C#","E"]
	#Microphone input exhibit
	f, amp = sample_sound(numNotes)
	f = np.asarray(np.reshape(f, (1, numNotes)))
	curve_arr = np.dot(f.T,(xdata+float(frame)))
	actual = np.zeros(j)
	for k in range(numNotes):
		curve = amp[k]*np.sin(curve_arr[k]+time.time())
		ax.plot(xdata[0], curve, label='{} (freq: {})'.format(MIDI_to_name(freq_to_MIDI(f[0,k])), str(int(f[0,k]))))
		ax.set_xlabel('Time')
		ax.set_ylabel('Amplitude')
		ax.set_title('Sound Components')
		actual +=  curve
	if (numNotes > 1):
		ax2.plot(xdata[0], actual, label="superposition")
		ax2.set_xlabel('Time')
		ax2.set_ylabel('Amplitude')
	ax.set_xlim(0, np.pi/50)
	ax.set_ylim(-np.max(amp), np.max(amp))
	ax2.set_xlim(0, np.pi/50)
	ax2.set_ylim(-2*np.max(amp), 2*np.max(amp))
	ax.legend(bbox_to_anchor=(1,1), bbox_transform=plt.gcf().transFigure)
	ax2.legend()
ani = FuncAnimation(fig, update, interval=10)
plt.show()
