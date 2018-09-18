import numpy as np
import pandas as pd

def window(n):
	midi_array = np.arange(n)
	f = 440.0*np.power(2,(midi_array-68.5)/12.)*total_time
	return f.astype(int)

#first 7 columns in the header are characteristics of the file and the rest are the signals
header = pd.read_csv('philharmonia_signals.csv',nrows=1,header=None).values

#the signals are labled by total_time so the last column has the total total_time
total_time = header[0,-1]

#the Hann window helps control the shape of the chopped up waveform by tappering around the edges
Hann= 0.5* (1-np.cos(np.linspace(0,2*np.pi, header.shape[1]-7,False)))	#end point not included

#total number of samples consist of chopped up files with given characteristics
characterstics = pd.read_csv('philharmonia_signals.csv',usecols=[0,1,2,3,4,5,6],dtype=str).values
total_rows = characterstics.shape[0]

#important characterstics for identifying useful samples (only normal non-phrases)
good_characteristics = [[],
						[],
						[],
						[],
						['0.5','0.25','1.0','1.5','long','very-long'],
						[],
						['normal','arco-normal','pizz-normal']]

#identify the useful locations that match the criteria above
useful_locations = np.array([True]*characterstics.shape[0])
for i in range(7):
	temp = np.array([False]*characterstics.shape[0])		
	if len(good_characteristics[i]) == 0:
		temp = ~temp
	for sample in good_characteristics[i]:
		temp = temp | (characterstics[:,i] == sample)
	useful_locations = useful_locations & temp

#the maximum MIDI number also represents the total number of features after resolution reduction
max_MIDI = 144

#gives the location of the upper bounds for all MIDI notes up until max_MIDI in the logarithmic scaling (Hertz)
top = window(max_MIDI)

#how many times to chunk the rows
num_chunks = 100

#number of rows in each chunk
partition = total_rows/num_chunks

#container for all the output FFTs after resolution reduction
FFT_samples = np.array([]).reshape(0,max_MIDI)

#need to hold onto this variable after getting out of scope for the remainder
curr_min_index = 0

#the number of useful samples to be stored
num_samples = np.sum(useful_locations)

for chunk in range(num_chunks+1):
	#lower bound for where the rows in current chunk start
	curr_min_index = partition*chunk
	#number of rows skipped increments as the chunk number increases
	skiprows = np.arange(curr_min_index)
	#read the signals from the current chunk
	curr_signals = pd.read_csv('philharmonia_signals.csv',nrows=partition,skiprows=skiprows, usecols=np.arange(7,header.shape[1])).values
	use = useful_locations[curr_min_index:(curr_min_index+partition)]
	#take the Fourier transform of the current signals with the Hann window function included 
	FFT = np.abs(np.fft.rfft(curr_signals[use]*Hann,axis=1))
	#adjust the FFT values so that they are in MIDI space and round noets within windows to the nearest MIDI
	adj_FFT = np.array([np.max(e,axis=1) if e.shape[1]>1 else np.zeros(e.shape[0]) for e in np.split(FFT,top,axis=1)[:-1]]).T
	#create new array of adjusted samples
	FFT_samples = np.vstack((FFT_samples, adj_FFT))
	#show progress
	print 'chunk #{}: {} of {}'.format(chunk, FFT_samples.shape[0], num_samples)

#save the FFT samples to a csv file to train on later
data_frame = pd.DataFrame(FFT_samples)
data_frame.to_csv('reduced_shape_FFT.csv')