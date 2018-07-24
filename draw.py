#! /usr/bin/env python
######################
#Author: Orlando Melchor-Alonso
#Last Updated: 2018/07/23
######################
#Objective: draw the harmonics of different instruments on a staff in real time
######################

import numpy as np
from numpy.linalg import norm
import pyaudio
from graphics import *
import sys
from sklearn.ensemble import RandomForestClassifier
import pickle
import matplotlib.pyplot as plt

#number of harmonics to read
numNotes = int(sys.argv[1])

#directories
model_dir = 'models/'
images_dir = 'images/'

#drawing parameters
N = 1000				      #set size of window

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

class Note:
	def __init__(self,frequency=0):
		#intialize to nothing drawn
		if frequency == 0:
			self.noteOval = Oval(Point(0,0), Point(0,0))
			self.noteLine = Line(Point(0,0), Point(0,0))
			self.sharpLV = Line(Point(0,0), Point(0,0))
			self.sharpRV = Line(Point(0,0), Point(0,0))
			self.sharpBH = Line(Point(0,0), Point(0,0))
			self.sharpTH = Line(Point(0,0), Point(0,0))
			self.extraLines = []
			return
		#get midi note, frequency, and actual note
		n0 = int(round(freq_to_MIDI(frequency)))
		octave = (n0 / 12) - 1
		noteName = n0 % 12
		#spacing between intervals on staff
		spacing = .024
		isSemiTone = noteName == 1 or noteName == 3 or noteName == 6 or noteName == 8 or noteName == 10
		interval = spacing*(5-(noteName-isSemiTone)/2-7*(octave-4)-(noteName-isSemiTone)%2)
		################note oval################
		noteOval = Oval(Point(N*(.51-.03),N*(.519+interval-.024)), Point(N*(.51+.03), N*(.519+interval+.024)))
		#change outline color depending on pitch accuracy
		c0 = num_cents(frequency)
		if c0 < -20:
			noteOval.setOutline('red')
		elif c0 > 20:
			noteOval.setOutline('blue')
		else:
			noteOval.setOutline('green')
		noteOval.setWidth(4)
		self.noteOval = noteOval
		################note vertical line################
		noteLine = Line(Point(0,0), Point(0,0))
		#above A#4 the quarter note line goes up
		if n0 > 70:
			noteLine = Line(Point(N*.482,N*(.5125+interval)), Point(N*.482, N*(.674+interval)))
		#otherwise the line goes down
		else:
			noteLine = Line(Point(N*.538,N*(.5125+interval)), Point(N*.538, N*(.351+interval)))
		noteLine.setWidth(8)
		self.noteLine = noteLine
		################sharp semitones################
		sharpLV = Line(Point(0,0), Point(0,0))
		sharpRV = Line(Point(0,0), Point(0,0))
		sharpBH = Line(Point(0,0), Point(0,0))
		sharpTH = Line(Point(0,0), Point(0,0))
		#if the note C#, D#, F#, G#, or A# add a sharp
		if isSemiTone:
			#left vertical line of sharp
			sharpLV = Line(Point(N*.4275,N*(.4575+interval)), Point(N*.4275, N*(.5825+interval)))
			sharpLV.setWidth(4)
			#right vertical line of sharp
			sharpRV = Line(Point(N*.4475,N*(.4525+interval)), Point(N*.4475, N*(.5775+interval)))
			sharpRV.setWidth(4)
			#bottom horizontal line of sharp
			sharpBH = Line(Point(N*.4125,N*(.55+interval)), Point(N*.4625, N*(.535+interval)))
			sharpBH.setWidth(10)
			#top horizontal line of sharp
			sharpTH = Line(Point(N*.4125,N*(.5025+interval)), Point(N*.4625, N*(.4875+interval)))
			sharpTH.setWidth(10)
		self.sharpLV = sharpLV
		self.sharpRV = sharpRV
		self.sharpBH = sharpBH
		self.sharpTH = sharpTH
		################lines below or above staff################
		extraLines = []
		#if note lies below staff add more lines downwards
		if n0 < 62:
			numLines = int(interval/.024-5)/2+1
			for itr in range(numLines):
				extraLine = Line(Point(N*(.51-.045),N*(.519+.024*(5+2*itr))), Point(N*(.51+.045),N*(.519+.024*(5+2*itr))))
				extraLine.setWidth(6)
				extraLines.append(extraLine)
		#if note lies above staff add more lines on top
		elif n0 > 80:
			numLines = -int(7+interval/.024)/2+1
			for itr in range(numLines):
				extraLine = Line(Point(N*(.51-.045),N*(.519+.024*(-7-2*itr))), Point(N*(.51+.045),N*(.519+.024*(-7-2*itr))))
				extraLine.setWidth(6)
				extraLines.append(extraLine)
		self.extraLines = extraLines
	def draw(self, win):
		self.noteOval.draw(win)
		self.noteLine.draw(win)
		self.sharpLV.draw(win)
		self.sharpRV.draw(win)
		self.sharpBH.draw(win)
		self.sharpTH.draw(win)
		for line in self.extraLines:	
			line.draw(win)
	def undraw(self):
		self.noteOval.undraw()
		self.noteLine.undraw()
		self.sharpLV.undraw()
		self.sharpRV.undraw()
		self.sharpBH.undraw()
		self.sharpTH.undraw()
		for line in self.extraLines:	
			line.undraw()

#returns float MIDI from float frequencies where decimals are the cents
def freq_to_MIDI(f):
	#midi = 69 is A4
	if f == 0:
		return 0
	return 69 + 12*np.log2(f/440.0)

#returns float frequency from float MIDI
def MIDI_to_freq(n):
	return 440 * pow(2.0,(n-69)/12.0)

#returns the cents
def num_cents(f1):
	n1=freq_to_MIDI(f1)
	n2=np.round(n1)
	return int(np.round((n1-n2),2)*100)

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
	FFT = np.abs(np.fft.rfft(signal))
	#filter above certain threshold and remove low and high pass
	filter_FFT(FFT)
	#determine harmonics
	f0 = harmonic_detector(FFT, n)
	#get all harmonics above the root
	f0_idx = model.predict([f0])
	f = f0[f0>=f0[f0_idx]]
	#pad the missing harmonics with zeros
	padding = f0.shape[0] - f.shape[0]
	f = np.pad(f, (0,padding), 'constant')
	return f

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
	for i in range(numHarm):
		#get the current peak
		curr_max = np.argmax(FFT)
		low_idx  = int(pow(2,-cents/1200.)*curr_max)
		high_idx = int(pow(2,cents/1200.)*curr_max)
		#exclude in and around current peak in future searches
		FFT[low_idx:high_idx] = 0
		#the harmonic is too high to be detected never too low
		harmonics[i] = curr_max
	#must resize harmonics because there are more samples with more Rec_secs
	return harmonics / Rec_secs

####################audio detection begins here####################

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

def main():
	#set N x N background
	win = GraphWin('Music Staff', N, N)
	win.setBackground('white')
	#draw staff
	staff = Image(Point(N*.5, N*.5), '{}treble.png'.format(images_dir))
	staff.draw(win)
	f = (np.zeros(numNotes)-1)
	notes = []
	#begin sampling sounds
	print 'you can now play...'
	while 1:
		#get top frequencies with note names
		f0 = f
		f = sample_sound(numNotes)
		#don't keep redrawing if the notes are the same
		if norm(f-f0) < 10:
			continue
		else:
			if len(notes)>0:
				for num in range(numNotes):
					notes[num].undraw()
					#del notes[num]
				notes = []
			#draw note on staff and note name with cents on margin
			for num in range(numNotes):
				note = Note(f[num])
				note.draw(win)
				notes.append(note)

main()