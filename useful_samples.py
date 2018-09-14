import numpy as np
import pandas as pd
import time as t

def freq_to_MIDI(f):
	return 69 + 12*np.log2(f/440.0)

def window(n,cents):
	f = 440.0*np.power(2,(n-69)/12.)
	high = (pow(2,cents/1200.)*f)*time
	return high.astype(int)

'''
adj_FFT = np.array([np.max(e,axis=1) if e.shape[1]!=0 else np.zeros(e.shape[0]) for e in np.split(FFT,top_windows,axis=1)[:-1]]).T


segments = pd.read_csv('philharmonia_signals.csv',usecols=[1]).values
start_locations = np.where(segments == 0)[0]

end_locations = np.zeros(start_locations.shape[0],int)
end_locations[:-1] = start_locations[1:]
end = segments.shape[0]
end_locations[-1] = end

useful_locations = start_locations+(end_locations - start_locations)/2
print useful_locations.shape[0]
'''


header = pd.read_csv('philharmonia_signals.csv',nrows=1,header=None).values
total_rows = pd.read_csv('philharmonia_signals.csv',usecols=[0]).values.shape[0]
time = header[0,-1]
max_MIDI = 144
cents = 50
top = window(np.arange(max_MIDI),cents)

num_chunks = 64
chunks = np.arange(num_chunks)
partition = total_rows/num_chunks
FFT_samples = np.zeros((total_rows,max_MIDI))
index = 0
# Hann window function ##https://en.wikipedia.org/wiki/Hann_function
Hann= 0.5* (1-np.cos(np.linspace(0,2*np.pi, header.shape[1]-7,False)))	#end point not included

for chunk in chunks:
	curr_min_index = partition*chunk
	skiprows = np.arange(curr_min_index)
	samples = pd.read_csv('philharmonia_signals.csv',nrows=partition,skiprows=skiprows, usecols=np.arange(7,header.shape[1])).values
	#locations = useful_locations<(curr_min_index+partition)
	#use = useful_locations[locations] - (curr_min_index)
	#useful_locations = useful_locations[~locations]
	FFT = np.abs(np.fft.rfft(samples*Hann,axis=1))
	adj_FFT = np.array([np.max(e,axis=1) if e.shape[1]>=1 else np.zeros(e.shape[0]) for e in np.split(FFT,top,axis=1)[:-1]]).T
	FFT_samples[index:(index+FFT.shape[0])] = adj_FFT
	index = index+FFT.shape[0]
	print '{} of {}'.format(index, total_rows)
data_frame = pd.DataFrame(FFT_samples)
data_frame.to_csv('reduced_shape_FFT.csv')