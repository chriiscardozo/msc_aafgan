from numpy import pi, sin
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import math

def raw_func_bhsa(x, t, l):
    h1 = 0.5 * math.sqrt((1 + 2*l*x)**2 + (4*(t**2)))
    h2 = 0.5 * math.sqrt((1 - 2*l*x)**2 + (4*(t**2)))

    # swish
    # return ((((math.e**(t * x) * (1 + math.e**(t * x) + t * x))/(1 + math.e**(t * x))**2))+0.1)/1.2

    return h1 - h2

def func_bh(x, t, l, interval_0_1=False):
    # if l > -0.001 and l < 0.001: l = 0.001
    result = raw_func_bhsa(x, t, l)
    if(interval_0_1): result = (result/2) + 0.5
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
t_0 = .5
l_0 = .5

z = np.arange(min_x, max_x, .01)


# Draw the initial plot
# The 'line' variable is used for modifying the line later
[line_01] = ax.plot(z, [func_bh(k, t_0, l_0, interval_0_1=True) for k in z], linewidth=1, color='blue')
[line_11] = ax.plot(z, [func_bh(k, t_0, l_0, interval_0_1=False) for k in z], linewidth=1, color='grey')
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
tau_slider_ax  = fig.add_axes([0.25, 0.15, 0.65, 0.03], facecolor=axis_color)
tau_slider = Slider(tau_slider_ax, 'tau', -15.0, 15.0, valinit=t_0)

l_slider_ax = fig.add_axes([0.25, 0.1, 0.65, 0.03], facecolor=axis_color)
l_slider = Slider(l_slider_ax, 'lambda', -15.0, 15.0, valinit=l_0)

# Define an action for modifying the line when any slider's value changes
def sliders_on_changed(val):
    line_01.set_ydata([func_bh(k, tau_slider.val, l_slider.val, interval_0_1=True) for k in z])
    line_11.set_ydata([func_bh(k, tau_slider.val, l_slider.val, interval_0_1=False) for k in z])
    fig.canvas.draw_idle()

tau_slider.on_changed(sliders_on_changed)
l_slider.on_changed(sliders_on_changed)


plt.show()