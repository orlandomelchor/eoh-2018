import numpy as np
import matplotlib.pyplot as plt
import math
from keras.datasets import mnist
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D
from keras.callbacks import TensorBoard
from keras.models import Model
from keras import backend as K
from keras.optimizers import SGD
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import LearningRateScheduler

def plotNumbers(original, noisy_original, reconstruction):
	m = 3
	n = 10
	plt.figure(figsize=(20, 2))
	for i in range(1,n*m+1):
	    ax = plt.subplot(m, n, i)
	    if i / 11 == 0:
		    plt.imshow(original[i%10].reshape((12,12)))
	    if i / 11 == 1:
		    plt.imshow(noisy_original[i%10].reshape((12,12)))
	    if i / 11 == 2:
		    plt.imshow(reconstruction[i%10].reshape((12,12)))
	    plt.gray()
	    ax.get_xaxis().set_visible(False)
	    ax.get_yaxis().set_visible(False)
	plt.show()


data = pd.read_csv('reduced_shape_FFT.csv').values[:,1:]
print data.shape
data =  MinMaxScaler().fit_transform(data)
print np.min(data[0])
print np.max(data[0])
np.random.shuffle(data)
size = int(data.shape[0]*.8)
x_train = data[:size]
x_test = data[size:]

print 'train shape: {}'.format(x_train.shape)
print 'test shape: {}'.format(x_test.shape)

size = x_train.shape[1]
#x_train = np.reshape(x_train, (len(x_train), size, 1))  # adapt this if using `channels_first` image data format
#x_test = np.reshape(x_test, (len(x_test), size, 1))  # adapt this if using `channels_first` image data format

noise_factor = 0.05
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)


#layer dimensions
input_dim = 144
layer1_dim = 72

#image
input_image = Input(shape=(input_dim,))

#layers
encoded = Dense(layer1_dim, activation='relu')(input_image)
print encoded.shape

decoded = Dense(input_dim, activation='sigmoid')(encoded)
print decoded.shape

#autoencoder models
autoencoder = Model(input_image, decoded)

#fit the autoencoder
sgd = SGD(lr=5, momentum=0.7, decay=0.0, nesterov=False)
autoencoder.compile(optimizer=sgd, loss='binary_crossentropy')
autoencoder.fit(x_train_noisy, x_train,
                epochs=1000,
                batch_size=1024,
                shuffle=True,
                validation_data=(x_test_noisy, x_test))

x_test_unnoisy = autoencoder.predict(x_test_noisy)

# serialize model to HDF5
autoencoder.save('test_network.h5')
print("Saved model to disk")
residual = x_test.flatten()-x_test_unnoisy.flatten()
plt.plot(residual)
plt.show()

plotNumbers(x_test, x_test_noisy, x_test_unnoisy)

