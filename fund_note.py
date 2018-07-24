#! /usr/bin/env python
######################
#Author: Orlando Melchor-Alonso
#Last Updated: 2018/07/23
######################
#Objective: create a csv file of the harmonics of different instruments.
#		    create a model for identifying the fundamental note in those harmonics  
######################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import time
import wave
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#number of harmonics
numHarm = int(sys.argv[1])

#parameters below here do not have to be powers of 2
min_freq = 29              #min detected frequency (lowest audible)
max_freq = 20000 	        #max detected frequency (highest audible)
cents    = 200		        #width of peak frequency in cents for harmonic detection
thresh   = 0.005              #percentage from peak frequency to use in threshold

#location of the samples, data, and output file
dir_ = 'philharmonia/all-samples/'
data_dir = 'philharmonia/data/'
output_csv = '{}philharmonia_harmonics_{}.csv'.format(data_dir, numHarm)

#location of saved models
classifier_dir = 'models/'

#all note names in instrument directories
noteNames = 'C Cs D Ds E F Fs G Gs A As B'.split(' ') 

#detects the peak frequencies in an FFT
def harmonic_detector(FFT, numHarm, rec):
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
	return harmonics / rec

#filters the FFT using from threshold and low/high pass filter
def filter(FFT, rec):
	#must first resize filters 
	low_pass  = int(min_freq*rec)
	high_pass = int(max_freq*rec)
	#low pass filter
	FFT[:low_pass] = 0
	#high paas filter
	FFT[high_pass:] = 0
	#remove noise within a certain percent of max
	FFT[FFT < np.max(FFT)*thresh] = 0
	
##########read the signal from a .wav file##########
def sample_sound_file(file, numHarm):
	spf = wave.open(file, 'r')
	signal = spf.readframes(-1)
	signal = np.fromstring(signal, 'Int16')

	#fft works faster for powers of 2 so pad the end with zeros
	padding = pow(2,int(np.log2(signal.shape[0]))+1)-signal.shape[0]
	signal = np.pad(signal, (0,padding), 'constant')

	Rec_secs = signal.shape[0]/float(spf.getframerate())

	#use real one dimensional fft and filter
	FFT = np.abs(np.fft.rfft(signal))
	filter(FFT, Rec_secs)

	#determine harmonics
	return harmonic_detector(FFT, numHarm, Rec_secs)

def freq_to_MIDI(f):
	#midi = 69 is A4
	fi = f[f!=0]
	ni = 69 + 12*np.log2(fi/440.0)
	n0 = f[f==0]
	return np.hstack((ni,n0))

def name_to_MIDI(note):
	return (noteNames.index(note[:-1]))+(int(note[-1])+1)*12

def MIDI_to_freq(n):
	return 440 * pow(2.0,(n-69)/12.0)

def find_nearest(n_harmonics, n):
    ci = 100*np.abs(n_harmonics - n)
    #print n_harmonics, n, ci, np.sum(ci < cents) < 1
    if np.sum(ci < cents) < 1:
    	return numHarm
    return (ci-cents).argmin()

def create_data():
	#column names for the csv file
	col = []
	for i in range(0, numHarm):
		col.append('harmonic #{}'.format(i))
	col.append('fundamental index')
	col.append('fundamental frequency')

	#initialize features, index labels, fundamental frequencies, and feature names
	X = np.array([]).reshape(0,numHarm)
	y = np.array([], int)
	f = np.array([], int)
	names = np.array([],str)

	#get the instrument directories
	directories = [instrument for instrument in os.listdir(dir_) if os.path.isdir(dir_+instrument)]
	for directory in directories:
		#print the instrument being loaded
		print '{}:'.format(directory)
		#get all the files for that instrument
		files = [name for name in os.listdir(dir_+directory+'/')]
		for file in files:
			#split the file by the '_' delimiter and remove the .wav extension 
			sep = file[:-4].split('_')
			#if the files only contain a single note and are not phrases do the following
			if (sep[4] == 'normal' and sep[2].isdigit()):
				#the features are the frequency harmonics from the fft of the signal
				features = sample_sound_file(dir_+directory+'/'+file, numHarm) #features
				#convert to midi form because musical notes are seperated logarithmically so we need to find nearest on a linear scale (MIDI)
				harmonics = freq_to_MIDI(features)
				n = int(name_to_MIDI(sep[1]))
				#the label is the index of the harmonics that is closest to the fundamental frequency
				label = find_nearest(harmonics, n)
				#print all the files that are excluded from the training (most are either too short, too soft, too low, or too high)
				if label == numHarm:
					print '\tremoving {}'.format(file)
					continue
				#build the features, labels, fundamental frequency, and feature names
				X = np.vstack((X,features))
				y = np.hstack((y, label))
				f = np.hstack((f, MIDI_to_freq(n)))
				names = np.hstack((names, file))
	print 'features shape: {}'.format(X.shape)
	print 'labels shape: {}'.format(y.shape)
	print 'frequency shape: {}'.format(f.shape)

	print 'saving csv file...'

	#save file of the data collected

	data_frame = np.hstack((X,y.reshape(y.shape[0],1),f.reshape(f.shape[0],1)))
	data = pd.DataFrame(data_frame, columns=col, index = names)
	data.to_csv(output_csv)
	return X, y

def read_data():
	data_frame = pd.read_csv(output_csv).values
	X = data_frame[:,1:-2]
	y = data_frame[:,-2].astype(int)
	f = data_frame[:,-1]
	print 'features shape: {}'.format(X.shape)
	print 'labels shape: {}'.format(y.shape)
	print 'frequency shape: {}'.format(f.shape)
	return X, y

def final_model(X,y):
	#fit RandomForest classifier on all the data
	model = RandomForestClassifier(n_estimators = 120, max_depth=None)
	print 'training...'
	model.fit(X, y)
	#save the model
	print 'saving model {}...'.format(numHarm)
	filename = 'fundamental_frequency_{}.sav'.format(numHarm)
	pickle.dump(model, open(classifier_dir+filename,'wb'))


def test_models(X,y):
	#split the data to see how well the classifier works on a subset of the data
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
	#because of the size of dataset and the fact that we have a labeled dataset use RandomForest
	model = RandomForestClassifier(n_estimators = 120, max_depth=None, random_state=1)
	print 'training...'
	model.fit(X_train, y_train)

	print 'testing...'
	y_pred = model.predict(X_test)

	#give the accuracy, number of files under each label, each unique label present, and the feature importance on each label
	print 'accuracy = {}%'.format(np.round(np.mean(y_pred == y_test)*100,3))
	classes = np.unique(y)
	print 'classes: {}'.format(classes)
	print 'bined labels: {}'.format(np.bincount(y))
	print 'feature importance: {}'.format(model.feature_importances_)

	#plot the confusion matrix, the more diagonal the better
	confusion = confusion_matrix(y_test, y_pred,labels=classes)
	plt.xlabel('prediction')
	plt.ylabel('true')
	plt.imshow(confusion)
	plt.show()

def main():
	if not os.path.isfile(output_csv):
		print '\nfile does not exist'
		print 'creating file...'
		X, y = create_data()
	elif sys.argv[2] == 'overwrite':
		X, y = create_data()
	elif sys.argv[2] == 'keep':
		X, y = read_data()
	else:
		return
	if sys.argv[3] == 'test':
		test_models(X,y)
	if sys.argv[3] == 'finalize':
		final_model(X,y)

main()