import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wave
import pyaudio
import os
import time
import csv
import dask.array as da

Format   = pyaudio.paInt16
Channels = 1
Chunk    = pow(2,-4)
#ideal file
phil_dir = 'philharmonia/all-samples/'
frame_rate = 44100.0
seconds = .125
length = int(frame_rate*seconds)
pad = False

def is_float(input):
  try:
    num = float(input)
  except ValueError:
    return False
  return True

def ideal_signal(file):
	spf = wave.open(file, 'r')
	frames = spf.readframes(-1)
	signal = np.fromstring(frames, 'Int16')
	if pad:	
		padding = length - signal.shape[0]%length
		signal = np.pad(signal, (0,padding), 'constant').reshape(-1,length)
	else:
		signal = signal[:(signal.shape[0]/length)*length].reshape(-1,length)
	return signal

def other(file):	
	audio = pyaudio.PyAudio()
	stream = audio.open(format=Format,
	                    channels=Channels,
	                    rate=Rate,
	                    input=True,
	                    frames_per_buffer=Chunk)
	#add noise
	print 'recording...'
	frames = []
	Rec_secs = (intv*length)/float(spf.getframerate())
	for i in range(0, int(float(Rate)/Chunk*Rec_secs)):
		data = stream.read(Chunk)
		frames.append(data)
	stream.stop_stream()
	stream.close()
	audio.terminate()
	frames = b''.join(frames)
	noise = np.fromstring(frames, 'Int16').reshape(intv,-1)
	noisy_signal = signal+noise
	return noisy_signal

segment = np.array([]).reshape(0,1)
info = np.array([]).reshape(0,5)
signal_data = np.array([]).reshape(0,length)
names = np.array([])
#get the instrument directories
out = open('philharmonia_signals.csv','w')
col = np.array(['','segment', 'instrument', 'note name', 'duration', 'dynamics', 'type'])
col = np.hstack((col, np.array(['{:0.5e}'.format(e) for e in np.linspace(0,length/44100.0,length)])))
out.write(",".join(col)+'\n')
directories = [instrument for instrument in os.listdir(phil_dir) if os.path.isdir(phil_dir+instrument)]
row_indx = 0
for itr, directory in enumerate(directories):
	#print the instrument being loaded
	print 'progress: {}%'.format(round(itr*100.0/len(directories),1))
	#get all the files for that instrument
	files = [name for name in os.listdir(phil_dir+directory+'/')]
	for file in files:
		#split the file by the '_' delimiter and remove the .wav extension 
		sep = file[:-4].split('_')
		if sep[4] == '':
			sep[4] = 'legato'
		if sep[2] == '05':
			sep[2] = '0.5'
		elif sep[2] == '025':
			sep[2] = '0.25'
		elif sep[2] == '1':
			sep[2] = '1.0'
		elif sep[2] == '15':
			sep[2] = '1.5'
		if (sep[4] == 'normal' and is_float(sep[2])):
			#if the files only contain a single note and are not phrases do the following
			#the features are the frequency harmonics from the fft of the signal
			signal = ideal_signal(phil_dir+directory+'/'+file)
			size = signal.shape[0]
			for i in range(signal.shape[0]):
				out.write('{},'.format(row_indx))
				out.write('{},'.format(i))
				for j in range(len(sep)):
					out.write('{},'.format(sep[j]))
				for j in range(signal.shape[1]-1):
					out.write('{},'.format(signal[i,j]))
				out.write(str(signal[i,-1])+'\n')
				row_indx+=1
	print row_indx
		#segment = np.vstack((segment,np.arange(size).reshape(-1,1)))
		#info = np.vstack((info,np.repeat(np.array(sep).reshape(1,-1),size,axis=0)))
		#signal_data = da.vstack((signal_data, signal))