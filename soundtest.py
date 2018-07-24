#! /usr/bin/env python
######################
#Authors: Anil Radhakrishnan
#         Sarah Habib
#         Orlando Melchor-Alonso
#Last Updated: 2018/03/20
######################
#Objective: Progam capable of identifying musical notes in real time
#Future: Create a visualization module
######################

import numpy as np 
import pyaudio
import matplotlib as pyplot
from matplotlib.animation import FuncAnimation

####
#https://newt.phys.unsw.edu.au/jw/notes.html resource for notes to frequency conversion
######################
#Sampling Parameters			##can be optimized

MIDIMin=50 #60	#C4
MIDIMax=90 #69	#A4				##chosen simply because they were prominent in resource 
FSamp=int(1e5)	#Hz	#Sampling rate for the FFT
FSize=2**10	#Frames per buffer
FFTSize=2**4	#Frames for FFT averaging

FFTSamp=int(FSize*FFTSize)	# Samples per FFR
dFreq= float(FSamp)/FFTSamp 	# Sampling resolution	

######################
#Music to math translation

noteNames='C C# D D# E F F# G G# A A# B'.split() # names for all the notes ## are there more?



def freq_to_MIDI(f): 
	'''Converts frequency to MIDI number'''
	return 69 + 12*np.log2(f/440.0)
def MIDI_to_freq(n):
	'''Converts MIDI number to frequency'''
	return 440 * 2.0**((n-69)/12.0)
def MIDI_name(n): 
	'''Converts MIDI number to note names'''
	return noteNames[int(n % 12)] + str(int(n/12) - 1)

def MIDI_to_bin(n):
	'''Converts MIDI number into FFT bin center frequency '''
	return MIDI_to_freq(n)/dFreq

#######################

# Allocate space to run an FFT. 
buf = np.zeros(FFTSamp, dtype=np.float32)		#sample buffer allocation
##global frames
##frames = 0

# Initialize audio
stream = pyaudio.PyAudio().open(format=pyaudio.paInt16,			# just a 16 bit integer
                                channels=1,
                                rate=FSamp,
                                input=True,
                                frames_per_buffer=int(FSize))

# Hann window function ##https://en.wikipedia.org/wiki/Hann_function
Hann= 0.5* (1-np.cos(np.linspace(0,2*np.pi, FFTSamp,False)))    #end point not included

# setting first and last FFt bin center frequency
imin = max(0, int(np.floor(MIDI_to_bin(MIDIMin-1))))
imax = min(FFTSamp, int(np.ceil(MIDI_to_bin(MIDIMax+1))))

# Print initial text
print('sampling at', FSamp, 'Hz with max resolution of', dFreq, 'Hz')
print()

#finds n number of harmonics
#spacing is the width of the peak
def peaks(arr, n, spacing=10):
    arr=np.array(arr)
    L=np.argsort(-1*arr)
    bad_idx=[]
    for i in range(len(L)-1):
        if abs(L[i]-L[i+1])<spacing:
            bad_idx.append(i+1)
    L=np.delete(L,bad_idx)
    peaks=L[:n]
    return peaks

def maximums(arr, initial, final):
    x_1 = arr[initial:final]
    b={np.abs(x_1[i]):(i+initial) for i in range(final-initial)}
    a = np.sort(x_1)[::-1]
    c = np.abs(a[0:3])
    return c

def sample_sound(num_freq):
    frames = 0
    stream.start_stream() #program starts listening 
    # While input stream is open
    while stream.is_active():

        # Shift the buffer down and new data in
        buf[:-FSize] = buf[FSize:]
        buf[-FSize:] = np.fromstring(stream.read(FSize, exception_on_overflow = False), np.int16)

        # Run the FFT on the windowed buffer
        signal = buf*Hann

        fourier = np.fft.rfft(signal)
        #print((np.abs(fourier[imin:imax]).argmax()+imin)*dFreq)
        fi = (peaks(np.abs(fourier[imin:imax]), num_freq) + imin) * dFreq
        #print(fourier)
        # Get frequency of maximum response in range
        #fi = np.array([((np.abs(fourier[imin:imax]).argmax() + imin) * dFreq)])

        # Get MIDI number and nearest int
        freq = []
        name = []
        for k in range(np.size(fi)):
            freq.append(fi[k])

            n = freq_to_MIDI(fi[k])
            n0 = int(round(n))
            name.append(MIDI_name(n0))
        
        # Console output once we have a full buffer
        ##global frames
        frames += 1

        if frames >= FFTSize:
            #print('frequency: {:8.0f} Hz     note: {:>3s} '.format(freq[0], MIDI_name(n0)))
            freq = np.sort(np.asarray(freq, float))
            stream.stop_stream()
            return freq, name