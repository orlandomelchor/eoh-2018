import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import pi, sin
#import soundtest as st

xdata = np.linspace(0, 2*np.pi, 60000)
j = np.size(xdata)

xdata = np.reshape(xdata, (1, j))
fig, ax = plt.subplots()

#ln, = plt.plot([], [], 'ro', animated=True)
def update(frame):
	ax.clear()
	#A Major Chord exhibit (A, C#, E)
	#f = np.array([440.0, 554.37, 659.25])
	#Microphone input exhibit
	f,name = st.sample_sound()
	f = np.array(f)

	i=np.size(f)
	f = np.reshape(f, (1, i))
	curve_arr = f.T @ (xdata+float(frame))
	actual = np.zeros(j)
	for k in range(i):
		curve = np.sin(curve_arr[k])
		ax.plot(xdata[0], curve)
		actual +=  curve
	if (i > 1):
		ax.plot(xdata[0], actual)
	ax.set_xlim(0, np.pi/50)
	ax.set_ylim(-5, 5)
#    ln.set_data(xdata, ydata)
#    return ln,

ani = FuncAnimation(fig, update, interval=50)
plt.show()