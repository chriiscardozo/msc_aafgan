from numpy import pi, sin
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import math

def raw_dsilu(x, b):
    # dSilu
    return ((((math.e**(b * x) * (1 + math.e**(b * x) + b * x))/(1 + math.e**(b * x))**2))+0.1)

def func_dsilu(x, b, interval_0_1=False):
    result = raw_dsilu(x, b)
    if(not interval_0_1): result = (result - 0.5) * 2
    return result


axis_color = 'lightgoldenrodyellow'
fig = plt.figure()
ax = fig.add_subplot(111)

# Adjust the subplots region to leave some space for the sliders and buttons
fig.subplots_adjust(left=0.25, bottom=0.25)

min_x = -10
max_x = 10
min_y = -2
max_y = 2
b_0 = .5

z = np.arange(min_x, max_x, .01)

# Draw the initial plot
# The 'line' variable is used for modifying the line later
[line_01] = ax.plot(z, [func_dsilu(k, b_0, interval_0_1=True) for k in z], linewidth=1, color='blue')
[line_cte1] = ax.plot(z, [1 for x in z], linewidth=1, color='red', linestyle='-')
[line_cte0] = ax.plot(z, [-1 for x in z], linewidth=1, color='red', linestyle='-')
[line_sigmoid] = ax.plot(z, [1/(1+np.exp(-x)) for x in z], linewidth=1, color='green', linestyle='dashed')
[line_tanh] = ax.plot(z, np.tanh(z), linewidth=1, color='green', linestyle='dotted')

ax.set_xlim([min_x, max_x])
ax.set_ylim([min_y, max_y])
ax.grid(True)
ax.set_xlabel('x')
ax.set_ylabel('y')

# Add sliders for tweaking the parameters
beta_slider_ax  = fig.add_axes([0.25, 0.15, 0.65, 0.03], facecolor=axis_color)
beta_slider = Slider(beta_slider_ax, 'beta', -15.0, 15.0, valinit=b_0)

# Define an action for modifying the line when any slider's value changes
def sliders_on_changed(val):
    line_01.set_ydata([func_dsilu(k, beta_slider.val, interval_0_1=True) for k in z])
    fig.canvas.draw_idle()

beta_slider.on_changed(sliders_on_changed)

plt.show()