import numpy as np 
import pyaudio
import matplotlib as pyplot
from matplotlib.animation import FuncAnimation

#! /usr/bin/env python
######################
#Author: Anil Radhakrishnan
#Last Updated: 2017/12/19
######################
#Objective: Progam capable of identifying musical notes in real time
#Future: Create a visualization module
######################

####
#https://newt.phys.unsw.edu.au/jw/notes.html resource for notes to frequency conversion
######################
#Sampling Parameters			##can be optimized

MIDIMin=60	#C4
MIDIMax=69	#A4				##chosen simply because they were prominent in resource 
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

def sample_sound():
    frames = 0
    stream.start_stream() #program starts listening 
    # While input stream is open
    while stream.is_active():

        # Shift the buffer down and new data in
        buf[:-FSize] = buf[FSize:]
        buf[-FSize:] = np.fromstring(stream.read(FSize, exception_on_overflow = False), np.int16)

        # Run the FFT on the windowed buffer
        fft = np.fft.rfft(buf * Hann)

        # Get frequency of maximum response in range
        freq = (np.abs(fft[imin:imax]).argmax() + imin) * dFreq

        # Get MIDI number and nearest int
        n = freq_to_MIDI(freq)
        n0 = int(round(n))

        # Console output once we have a full buffer
        ##global frames
        frames += 1

        if frames >= FFTSize:
            print('frequency: {:8.0f} Hz     note: {:>3s} '.format(freq, MIDI_name(n0)))
            stream.stop_stream()
            return freq