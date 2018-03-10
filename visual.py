import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math

fig, ax = plt.subplots()
xdata = np.linspace(0, 2*np.pi, 200)
ydata = np.sin(xdata)
ax.plot(xdata, ydata)
#ln, = plt.plot([], [], 'ro', animated=True)

def update(frame):
	ydata = np.sin(xdata + frame)
	ax.clear()
	ax.plot(xdata, ydata)
#    ln.set_data(xdata, ydata)
#    return ln,

ani = FuncAnimation(fig, update)
plt.show()