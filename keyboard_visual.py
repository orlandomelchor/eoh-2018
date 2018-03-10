import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import pi, sin
import soundtest as st

xdata = np.linspace(0, 2*np.pi, 30000)
j = np.size(xdata)

xdata = np.reshape(xdata, (1, j))
fig = plt.figure()
ax = fig.add_subplot(211)
ax2 = fig.add_subplot(212)



def update(frame):
	ax.clear()
	ax2.clear()
	#A Major Chord exhibit (A, C#, E)
	#f, name = np.asarray([440.0, 554.37, 659.25]), ["A","C#","E"]
	#Microphone input exhibit
	f, name = st.sample_sound()
	f = np.asarray(f)

	i=np.size(f)
	f = np.reshape(f, (1, i))
	curve_arr = f.T @ (xdata+float(frame))
	actual = np.zeros(j)
	for k in range(i):
		curve = np.sin(curve_arr[k])
		ax.plot(xdata[0], curve, label=name[k] + " (freq: " + str(int(f[0,k])) + ")")
		ax.set_xlabel('Time')
		ax.set_ylabel('Amplitude')
		ax.set_title('Sound Components')
		actual +=  curve
	if (i > 1):
		ax2.plot(xdata[0], actual, label="superposition")
		ax2.set_xlabel('Time')
		ax2.set_ylabel('Amplitude')
	ax.set_xlim(0, np.pi/50)
	ax.set_ylim(-5, 5)
	ax2.set_xlim(0, np.pi/50)
	ax2.set_ylim(-10, 10)
	ax.legend(bbox_to_anchor=(1,1), bbox_transform=plt.gcf().transFigure)
	ax2.legend()

ani = FuncAnimation(fig, update, interval=10)
plt.show()