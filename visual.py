import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import soundtest as st

fig, ax = plt.subplots()
xdata = np.linspace(0, 2*np.pi, 10000)
y1data = np.sin(xdata)
ax.plot(xdata, y1data)
#ln, = plt.plot([], [], 'ro', animated=True)

def update(frame):
	f1, name = st.sample_sound()
	y1data = np.sin(f1*xdata + frame)
	ax.clear()
	ax.plot(xdata, y1data)
	ax.set_ylim(-1.5,1.5)
	ax.set_xlim(0, np.pi/10)
#    ln.set_data(xdata, ydata)
#    return ln,

ani = FuncAnimation(fig, update, interval=50)

plt.show()
