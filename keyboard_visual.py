import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import pi, sin
#import soundtest as st

xdata = np.linspace(0, 2*np.pi, 60000)
j = np.size(xdata)

xdata = np.reshape(xdata, (1, j))
fig, ax = plt.subplots()

def update(frame):
	ax.clear()
	#A Major Chord exhibit (A, C#, E)
	f, name = np.asarray([440.0, 554.37, 659.25]), ["A","C#","E"]
	#Microphone input exhibit
	#f, name = st.sample_sound()
	f = np.asarray(f)

	i=np.size(f)
	f = np.reshape(f, (1, i))
	curve_arr = f.T @ (xdata+float(frame))
	actual = np.zeros(j)
	for k in range(i):
		curve = np.sin(curve_arr[k])
		ax.plot(xdata[0], curve, label=name[k] + " (freq: " + str(int(f[0,k])) + ")")
		actual +=  curve
	if (i > 1):
		ax.plot(xdata[0], actual)
		ax.plot(xdata[0], actual, label="superposition")
	ax.set_xlim(0, np.pi/50)
	ax.set_ylim(-5, 5)
	plt.legend()

ani = FuncAnimation(fig, update, interval=50)
plt.show()