import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import soundtest as st

fig, ax = plt.subplots()
xdata = np.linspace(0, 2*np.pi, 200)
ydata = np.sin(xdata)
ax.plot(xdata, ydata)
#ln, = plt.plot([], [], 'ro', animated=True)

def update(frame):
	f = st.sample_sound()
	ydata = np.sin(f*xdata + frame)
	ax.clear()
	ax.plot(xdata, ydata)
#    ln.set_data(xdata, ydata)
#    return ln,

ani = FuncAnimation(fig, update, interval=50)
plt.show()