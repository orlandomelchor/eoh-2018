import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import pi, sin

#f1 = np.array([440.0, 554.37, 659.25]) #The notes are A, C#, E (A Major Chord)
xdata = np.linspace(0, 2*np.pi, 10000)
i,j = np.size(f1), np.size(xdata)

f1 = np.reshape(f1, (1, i))
xdata = np.reshape(xdata, (1, j))
fig, ax = plt.subplots()

#ln, = plt.plot([], [], 'ro', animated=True)
def update(frame):
	ax.clear()
	f1 = st.sample_sound()
	curve_arr = f1.T @ (xdata+float(frame))
	actual = np.zeros(j)
	for k in range(i):
		curve = np.sin(curve_arr[k])
		ax.plot(xdata[0], curve)
		actual +=  curve
	ax.plot(xdata[0], actual)
	ax.set_xlim(0, np.pi/50)
	ax.set_ylim(-5, 5)
#    ln.set_data(xdata, ydata)
#    return ln,

ani = FuncAnimation(fig, update)
plt.show()