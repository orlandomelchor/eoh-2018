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
numHarm = 0
if len(sys.argv)>3:
	numHarm = int(sys.argv[3])

#parameters below here do not have to be powers of 2
min_freq = 29              #min detected frequency (lowest audible)
max_freq = 20000 	        #max detected frequency (highest audible)
cents    = 200		        #width of peak frequency in cents for harmonic detection
thresh   = 0.005              #percentage from peak frequency to use in threshold

#location of the samples, data, and output file
dir_ = 'philharmonia/all-samples/'
data_dir = 'philharmonia/data/'
output_csv = '{}philharmonia_harmonics.csv'.format(data_dir)

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
	
#read the signal from a .wav 
def sample_sound_file(file, numHarm):
	#load signal
	spf = wave.open(file, 'r')
	signal = spf.readframes(-1)
	signal = np.fromstring(signal, 'Int16')
	#fft works faster for powers of 2 so pad the end with zeros
	padding = pow(2,int(np.log2(signal.shape[0]))+1)-signal.shape[0]
	signal = np.pad(signal, (0,padding), 'constant')
	#number of recorded seconds
	Rec_secs = signal.shape[0]/float(spf.getframerate())
	#use real one dimensional fft and filter
	FFT = np.abs(np.fft.rfft(signal))
	filter(FFT, Rec_secs)
	#determine harmonics
	return harmonic_detector(FFT, numHarm, Rec_secs)

#convert frequency to MIDI
def freq_to_MIDI(f):
	#don't want to screw around with the original
	f2 = np.copy(f)
	#if any zeros are present just leave them alone otherwise convert frequencies to MIDI
	f2[f2!=0] = 69 + 12*np.log2(f2[f2!=0]/440.0)
	return f2

#convert a note name to a MIDI note
def name_to_MIDI(note):
	return (noteNames.index(note[:-1]))+(int(note[-1])+1)*12

#convert MIDI note to a frequency
def MIDI_to_freq(n):
	return 440 * pow(2.0,(n-69)/12.0)

#find the index closest to the fundamental in a list of harmonics
def find_nearest(n_harmonics, n):
    ci = 100*np.abs(n_harmonics - n.reshape(n.shape[0],-1))
    #get the elements without a nearest value within the cents threshold
    cond = (np.sum(ci < cents, axis=1) == 0)
    labels = np.zeros(cond.shape[0],int)
    #label samples without a fundamental present in the harmonics as numHarm 
    labels[cond] = numHarm
    #otherwise use the location of the closest harmonic to the fundamental
    labels[~cond] = np.argmin((ci[~cond]-cents),axis=1)
    return labels

#write the philharmonia csv file
def write_data():
	print 'writing data...'
	totHarm = 1000
	#column names for the csv file
	col = []
	for i in range(0, totHarm):
		col.append('harmonic #{}'.format(i))
	col.append('fundamental frequency')
	#initialize features, index labels, fundamental frequencies, and feature names
	X = np.array([]).reshape(0,totHarm)
	f = np.array([], int)
	names = np.array([],str)
	#get the instrument directories
	directories = [instrument for instrument in os.listdir(dir_) if os.path.isdir(dir_+instrument)]
	for itr, directory in enumerate(directories):
		#print the instrument being loaded
		print 'progress: {}%'.format(round(itr*100.0/len(directories),1))
		#get all the files for that instrument
		files = [name for name in os.listdir(dir_+directory+'/')]
		for file in files:
			#split the file by the '_' delimiter and remove the .wav extension 
			sep = file[:-4].split('_')
			#if the files only contain a single note and are not phrases do the following
			if (sep[4] == 'normal' and sep[2].isdigit()):
				#the features are the frequency harmonics from the fft of the signal
				features = sample_sound_file(dir_+directory+'/'+file, totHarm) #features
				n = int(name_to_MIDI(sep[1]))
				#build the features, labels, fundamental frequency, and feature names
				X = np.vstack((X,features))
				f = np.hstack((f, MIDI_to_freq(n)))
				names = np.hstack((names, file))
	print 'progress: 100.0%'
	#shapes just for check
	print 'features shape: {}'.format(X.shape)
	print 'frequency shape: {}'.format(f.shape)
	#save file of the data collected
	print 'saving csv file...'
	data_frame = np.hstack((X,f.reshape(f.shape[0],1)))
	data = pd.DataFrame(data_frame, columns=col, index = names)
	data.to_csv(output_csv)

def read_data():
	print 'loading csv file...'
	#get the column names
	col = pd.read_csv(output_csv, nrows=1).columns
	#get the row names
	names = pd.read_csv(output_csv, usecols=[0]).values.T[0]
	#all of the features
	X = pd.read_csv(output_csv, usecols=col[1:numHarm+1]).values.astype(float)
	#the fundamental frequency
	f = pd.read_csv(output_csv, usecols=[col[-1]]).values.astype(float)
	#convert to MIDI because nearest values should be done linearly not logarithmically
	harmonics = freq_to_MIDI(X)
	n = freq_to_MIDI(f)
	#the label is the index of the harmonics that is closest to the fundamental frequency
	y = find_nearest(harmonics, np.round(n))
	#omitted samples
	omit = names[y==numHarm]
	for i in range(omit.shape[0]):
		print 'omitting {}'.format(omit[i])
	#only keep the non omitted samples
	X = X[y!=numHarm]
	y = y[y!=numHarm]
	#feature and label shapes for sanity check
	print 'features shape: {}'.format(X.shape)
	print 'labels shape: {}'.format(y.shape)
	return X, y

def final_model(X,y):
	#fit RandomForest classifier on all of the data
	model = RandomForestClassifier(n_estimators = 120, max_depth=None)
	print 'training...'
	model.fit(X, y)
	#save the model
	print 'saving model {}...'.format(numHarm)
	filename = 'fundamental_frequency_{}.sav'.format(numHarm)
	pickle.dump(model, open(classifier_dir+filename,'wb'))


def test_models(X,y):
	#split the data to test a subset of the data
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
	#because of the size of dataset and because it is labeled use RandomForest
	model = RandomForestClassifier(n_estimators = 120, max_depth=None, random_state=1)
	print 'training...'
	model.fit(X_train, y_train)
	print 'testing...'
	y_pred = model.predict(X_test)
	#give the accuracy, number of files under each unique label, and the feature importance on each label
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
	#option to write data
	if sys.argv[1] == 'write':
		write_data()
		return
	#option to raed data stored in the csv file
	elif sys.argv[1] == 'read':
		#if the file does not already exist create it first
		if not os.path.isfile(output_csv):
			write_data()
		#read the csv file
		X, y = read_data()
		#available if you want to try different ML approaches
		if sys.argv[2] == 'test':
			test_models(X,y)
		#finalize the model
		elif sys.argv[2] == 'finalize':
			final_model(X,y)

main()