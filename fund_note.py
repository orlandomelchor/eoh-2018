'''
Author: Orlando Melchor-Alonso
Last Updated: 2018/07/23

Objective: create a csv file of the harmonics of different instruments.
		   create a model for identifying the fundamental note in those harmonics
'''
import os
import sys
import wave
import pickle
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# number of harmonics
TOTAL_HARMONICS = 1000
HARMONIC_NUMBER = 0
if len(sys.argv) > 3:
    HARMONIC_NUMBER = int(sys.argv[3])

# parameters below here do not have to be powers of 2
MIN_FREQ = 29  # min detected frequency (lowest audible)
MAX_FREQ = 20000  # max detected frequency (highest audible)
CENTS = 200  # width of peak frequency in cents for harmonic detection
THRESH = 0.005  # percentage from peak frequency to use in threshold

# location of the samples, data, and output file
PHIL_DIR = 'philharmonia/all-samples/'
GOOD_DIR = '/media/orlandomelchor/My Passport/datasets/good-sound/good-sounds/'
DATA_DIR = 'data/'
OUTPUT_CSV = '{}harmonics.csv'.format(DATA_DIR)

# location of saved models
CLASSIFIER_DIR = 'models/'

# all note names in instrument directories
NOTE_NAMES = 'C Cs D Ds E F Fs G Gs A As B'.split(' ')

def harmonic_detector(fft, harmonic_number, rec):
    '''Method for detecting the harmonics in a signal

    Inputs: fft, harmonic_number, rec
            fft (array): The fourier transformed signal.
            harmonic_number (int): The number of harmonics to select from the signal.
            rec (float): Normalizing value for the fourier transform.

    Output:
            harmonics (array): The list of harmonics detected in the signal
            '''
    harmonics = np.zeros(harmonic_number)
    for i in range(harmonic_number):
        # get the current peak
        curr_max = np.argmax(fft)
        low_idx = int(pow(2, -CENTS / 1200.) * curr_max)
        high_idx = int(pow(2, CENTS / 1200.) * curr_max)
        # exclude in and around current peak in future searches
        fft[low_idx:high_idx] = 0
        # the harmonic is too high to be detected never too low
        harmonics[i] = curr_max
    # must resize harmonics because there are more samples with more rec_secs
    return harmonics / rec

# filters the FFT using from threshold and low/high pass filter


def fft_filter(fft, rec):
    '''Low-pass, high-pass and noise reduction Filters for the signal

    Inputs: fft, rec
            fft (array): The fourier transformed signal.
            rec (float): Normalizing value for the fourier transform.
    '''
    # must first resize filters
    low_pass = int(MIN_FREQ * rec)
    high_pass = int(MAX_FREQ * rec)
    # low pass filter
    fft[:low_pass] = 0
    # high paas filter
    fft[high_pass:] = 0
    # remove noise within a certain percent of max
    fft[fft < np.max(fft) * THRESH] = 0


def sample_sound_file(file, harmonic_number):
    '''Method for reading the signal from a .wav file.

    Inputs: file, harmonic_number
            file (string): The file name of the signal to load.
            harmonic_number (int): The number of harmonics to select from the signal.

    Outputs: harmonics
            harmonics (array): The list of harmonics detected in the signal
    '''
    # load signal
    spf = wave.open(file, 'r')
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, 'Int16')
    # fft works faster for powers of 2 so pad the end with zeros
    padding = pow(2, int(np.log2(signal.shape[0])) + 1) - signal.shape[0]
    signal = np.pad(signal, (0, padding), 'constant')
    # number of recorded seconds
    rec_secs = signal.shape[0] / float(spf.getframerate())
    # use real one dimensional fft and filter
    fft = np.abs(np.fft.rfft(signal))
    fft_filter(fft, rec_secs)
    # determine harmonics
    harmonics = harmonic_detector(fft, harmonic_number, rec_secs)
    return harmonics


def freq_to_MIDI(f):
    '''Method used to convert frequency values to MIDI.

    Inputs: f
            f (float array): Frequency values to convert to nearest MIDI values

    Outputs: f_copy
            f_copy (float array): MIDI values corresponding to the given ferquencies
    '''
    # don't want to screw around with the original
    f_copy = np.copy(f)
    # if any zeros are present just leave them alone otherwise convert
    # frequencies to MIDI
    f_copy[f_copy != 0] = 69 + 12 * np.log2(f_copy[f_copy != 0] / 440.0)
    return f_copy


def name_to_MIDI(note):
    '''Method used to convert a note name to a MIDI note.

    Inputs: note
            note (string): Note name to convert to a MIDI note.

    Outputs: name
            midi (int): The midi value corresponding to the given note name.
    '''
    midi = (NOTE_NAMES.index(note[:-1])) + (int(note[-1]) + 1) * 12
    return midi


def name_to_freq(note):
    '''Method used to convert a note name to a frequency value.

    Inputs: note
            note (string): Note name to convert to a frequency value.

    Outputs: name
            midi (int): The frequency value corresponding to the given note name.
    '''
    return MIDI_to_freq(name_to_MIDI(note))


def MIDI_to_freq(midi):
    '''Method used to convert a MIDI note to a frequency value.

    Inputs: midi
            midi (int array): MIDI notes to convert to frequency values.

    Outputs: freq
            freq (float array): The frequency values corresponding to the given MIDI notes.
    '''
    freq = 440 * pow(2.0, (midi - 69) / 12.0)
    return freq


def find_nearest(n_harmonics, midi):
    ''' Method for finding the index closest to the fundamental in a list of harmonics.

    Inputs: n_harmonics, midi
            n_harmonics (array): list of harmonics.
            midi (int): number of harmonics to find.

    Outputs: labels
            labels (int): Location of the closest harmonic to the funamental.
    '''
    c_i = 100 * np.abs(n_harmonics - midi.reshape(midi.shape[0], -1))
    # get the elements without a nearest value within the CENTS threshold
    cond = (np.sum(c_i < CENTS, axis=1) == 0)
    labels = np.zeros(cond.shape[0], int)
    # label samples without a fundamental present in the harmonics as
    # HARMONIC_NUMBER
    labels[cond] = HARMONIC_NUMBER
    # otherwise use the location of the closest harmonic to the fundamental
    labels[~cond] = np.argmin((c_i[~cond] - CENTS), axis=1)
    return labels


def write_data():
    '''Method for writing the philharmonia dataset to a csv file'''
    # column names for the csv file
    col = []
    for i in range(0, TOTAL_HARMONICS):
        col.append('harmonic #{}'.format(i))
    col.append('fundamental frequency')
    # good_names, good_data = good_sound_data()
    phil_names, phil_data = philharmonia_data()
    names = phil_names
    # names = np.hstack((phil_names, good_names))
    data_frame = phil_data
    # data_frame = np.hstack((phil_data, good_data))
    data = pd.DataFrame(data_frame, columns=col, index=names)
    data.to_csv(OUTPUT_CSV)


def get_table(table_name, conn, columns):
    '''Method for finding a table in a SQL dataset

    Inputs: table_name, conn, columns
            table_name (string): Name of the table to retrieve.
            conn (object): Connection cursor for the SQL dataset.
            columns (int): Columns to retrieve from the table.

    Outputs:
            data (array): The data stored in that table at those columns.
    '''
    prop = ",".join(columns)
    conn.execute('SELECT {} FROM {}'.format(prop, table_name))
    data = np.array(conn.fetchall(), dtype=object)
    return data


def good_sound_data():
    '''Method for retrieving data from the Good Sound dataset stored in a SQL file.

    Inputs: void

    Ouputs: names, data_frame
            names (string): The name of the files.
            data_frame (string): The data stored in those files.
    '''
    print('loading good sounds...')
    # connect to the good_sounds database
    sqlite_file = 'database.sqlite'
    conn = sqlite3.connect(sqlite_file)
    conn = conn.cursor()

    ###########################sound data###########################
    # gets note index, note names, and quality
    sound_cols = np.array([u'id', u'note', u'octave', u'klass'])
    sound_table = get_table('sounds', conn, sound_cols)

    # get all of the unique sound labels exluding None
    sound_types = np.unique(sound_table[:, 3])

    # if it is not this it is a valid sound type
    sound_types = np.array([sound_type for sound_type in sound_types
                            if sound_type is not None
                            if sound_type != ''
                            if not 'scale' in sound_type
                            if not 'pitch' in sound_type
                            if not 'tremolo' in sound_type
                            if not 'bad-attack' in sound_type
                            if not 'timbre-errors' in sound_type
                            if not 'stability-timbre' in sound_type])

    # get the elements that are valid sound types and replace the previous
    # sound_table with this one
    valid_samples = sound_table[:, 3] == 'good-sound'
    for good in sound_types:
        valid_samples = valid_samples | (sound_table[:, 3] == good)
    sound_table = sound_table[valid_samples, :]

    # combine the note name and the octave together to form the note replacing
    # a sharp with an s
    notes = np.array([w.replace('#', 's')
                      for w in sound_table[:, 1] + sound_table[:, 2].astype(str)])

    # form a sound_table with the notes instead of the note name and the octave
    sound_table = np.hstack(
        (sound_table[:, 0].reshape(-1, 1), notes.reshape(-1, 1)))

    ###########################file data###########################
    print('loading file names...')
    # get note id and file names
    takes_cols = np.array([u'id', u'filename'])
    takes_table = get_table('takes', conn, takes_cols)
    conn.close()

    # set a blank array using the 'sound' dimension and the 'takes' dimension
    # without the indices labeled
    data = np.array([]).reshape(0, sound_table.shape[
        1] + takes_table.shape[1] - 1)
    for sound_sample in sound_table:
        # samples repeat because recording devices change for the same samples
        # so we need those as well
        files_info = takes_table[takes_table[:, 0] == sound_sample[0]]
        # the note names are the same for the repeating samples
        note_names = np.tile(sound_sample[1:], (files_info.shape[0], 1))
        # data will consist of id, filename, and note name
        repeated_samples = np.hstack((files_info, note_names))
        data = np.vstack((data, repeated_samples))
    data = data[:, 1:]

    ###########################fft samples###########################
    print('writing good sounds data...')
    # initialize features, index labels, fundamental frequencies, and feature
    # names
    X = np.array([]).reshape(0, TOTAL_HARMONICS)
    f = np.array([], int)
    names = np.array([], str)

    itr = 0
    for file, note_name in data:
        # if (itr%30 == 1):
        #	print('progress: {}%'.format(round(itr*100.0/data.shape[0],1)))
        # the features are the frequency harmonics from the fft of the signal
        features = sample_sound_file(
            GOOD_DIR + file, TOTAL_HARMONICS)  # features
        midi = int(name_to_MIDI(note_name))
        # build the features, labels, fundamental frequency, and feature names
        X = np.vstack((X, features))
        f = np.hstack((f, MIDI_to_freq(midi)))
        names = np.hstack((names, file))
        itr += 1
    # shapes just for check
    print('features shape: {}'.format(X.shape))
    print('frequency shape: {}'.format(f.shape))
    print('writing data frame...')
    data_frame = np.hstack((X, f.reshape(f.shape[0], 1)))
    return names, data_frame


def philharmonia_data():
    '''Method for writing the philharmonia dataset to a csv file.

    Inputs: void

    Outputs: names, data_frame
            names (string): The names of the files written.
            data_farme (array): The data stored in those files.
    '''
    print('writing philharmonia data...')
    # initialize features, index labels, fundamental frequencies, and feature
    # names
    X = np.array([]).reshape(0, TOTAL_HARMONICS)
    f = np.array([], int)
    names = np.array([], str)
    # get the instrument directories
    directories = [instrument for instrument in os.listdir(
        PHIL_DIR) if os.path.isdir(PHIL_DIR + instrument)]
    for itr, directory in enumerate(directories):
        # print the instrument being loaded
        print('progress: {}%'.format(
            round(itr * 100.0 / len(directories), 1)), end='\r')
        # get all the files for that instrument
        files = [name for name in os.listdir(PHIL_DIR + directory + '/')]
        for file in files:
            # split the file by the '_' delimiter and remove the .wav extension
            sep = file[:-4].split('_')
            # if the files only contain a single note and are not phrases do
            # the following
            if (sep[4] == 'normal' and sep[2].isdigit()):
                # the features are the frequency harmonics from the fft of the
                # signal
                features = sample_sound_file(
                    PHIL_DIR + directory + '/' + file, TOTAL_HARMONICS)  # features
                midi = int(name_to_MIDI(sep[1]))
                # build the features, labels, fundamental frequency, and
                # feature names
                X = np.vstack((X, features))
                f = np.hstack((f, MIDI_to_freq(midi)))
                names = np.hstack((names, file))
    print('progress: 100.0%')
    # shapes just for check
    print('features shape: {}'.format(X.shape))
    print('frequency shape: {}'.format(f.shape))
    # save file of the data collected
    print('writing data frame...')
    data_frame = np.hstack((X, f.reshape(f.shape[0], 1)))
    return names, data_frame


def read_data():
    '''Method for reading the data from disk
    Inputs: void

    Outputs: X, y
            X (array): Data storage container.
            y (array): Labels storage container.
    '''
    print('loading csv file...')
    # get the column names
    col = pd.read_csv(OUTPUT_CSV, nrows=1).columns
    # get the row names
    names = pd.read_csv(OUTPUT_CSV, usecols=[0]).values.T[0]
    # all of the features
    X = pd.read_csv(OUTPUT_CSV, usecols=col[
        1:HARMONIC_NUMBER + 1]).values.astype(float)
    # the fundamental frequency
    f = pd.read_csv(OUTPUT_CSV, usecols=[col[-1]]).values.astype(float)
    # convert to MIDI because nearest values should be done linearly not
    # logarithmically
    harmonics = freq_to_MIDI(X)
    midi = freq_to_MIDI(f)
    # the label is the index of the harmonics that is closest to the
    # fundamental frequency
    y = find_nearest(harmonics, np.round(midi))
    # omitted samples
    omit = names[y == HARMONIC_NUMBER]
    for i in range(omit.shape[0]):
        print('omitting {}'.format(omit[i]))
    # only keep the non omitted samples
    X = X[y != HARMONIC_NUMBER]
    y = y[y != HARMONIC_NUMBER]
    # feature and label shapes for sanity check
    print('features shape: {}'.format(X.shape))
    print('labels shape: {}'.format(y.shape))
    return X, y


def final_model(X, y):
    '''Method for training the final model for the harmonics detector'''
    # fit RandomForest classifier on all of the data
    model = RandomForestClassifier(n_estimators=120, max_depth=None)
    print('training...')
    model.fit(X, y)
    # save the model
    print('saving model {}...'.format(HARMONIC_NUMBER))
    filename = 'fundamental_frequency_{}.sav'.format(HARMONIC_NUMBER)
    pickle.dump(model, open(CLASSIFIER_DIR + filename, 'wb'))


def test_models(X, y):
    '''Method for development purposes and testing'''
    # split the data to test a subset of the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1)
    # because of the size of dataset and because it is labeled use RandomForest
    model = RandomForestClassifier(
        n_estimators=120, max_depth=None, random_state=1)
    print('training...')
    model.fit(X_train, y_train)
    print('testing...')
    y_pred = model.predict(X_test)
    # give the accuracy, number of files under each unique label, and the
    # feature importance on each label
    print('accuracy = {}%'.format(np.round(np.mean(y_pred == y_test) * 100, 3)))
    classes = np.unique(y)
    print('classes: {}'.format(classes))
    print('bined labels: {}'.format(np.bincount(y)))
    print('feature importance: {}'.format(model.feature_importances_))
    # plot the confusion matrix, the more diagonal the better
    confusion = confusion_matrix(y_test, y_pred, labels=classes)
    plt.xlabel('prediction')
    plt.ylabel('true')
    plt.imshow(confusion)
    plt.show()


def main():
    # option to write data
    if sys.argv[1] == 'write':
        write_data()
        return
    # option to raed data stored in the csv file
    elif sys.argv[1] == 'read':
        # if the file does not already exist create it first
        if not os.path.isfile(OUTPUT_CSV):
            write_data()
        # read the csv file
        X, y = read_data()
        # available if you want to try different ML approaches
        if sys.argv[2] == 'test':
            test_models(X, y)
        # finalize the model
        elif sys.argv[2] == 'finalize':
            final_model(X, y)

main()
