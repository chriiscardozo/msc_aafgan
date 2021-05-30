from numpy import pi, sin
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import math


def func_mish(x, interval_0_1=False):
    return x * np.tanh(np.log(1 + np.exp(x)))


def func_mida(x, a, b, interval_0_1=False):
    return a * x * np.tanh(np.log(1 + np.exp(x + b)))


def raw_func_bhsa(x, t, l):
    h1 = 0.5 * math.sqrt((1 + 2*l*x)**2 + (4*(t**2)))
    h2 = 0.5 * math.sqrt((1 - 2*l*x)**2 + (4*(t**2)))
    return h1 - h2


def func_midab(x, t, l, interval_0_1=False):
    param = np.log(1 + np.exp(x))
    return x * raw_func_bhsa(param, t, l*l)


axis_color = 'lightgoldenrodyellow'
fig = plt.figure()
ax = fig.add_subplot(111)

# Adjust the subplots region to leave some space for the sliders and buttons
fig.subplots_adjust(left=0.25, bottom=0.25)

min_x = -5
max_x = 5
min_y = -5
max_y = 5
l0 = 1
t0 = 1

z = np.arange(min_x, max_x, .01)

# Draw the initial plot
# The 'line' variable is used for modifying the line later
# [line_01] = ax.plot(z, [func_mila(k, b_0, interval_0_1=True) for k in z], linewidth=1, color='blue')
[line_02] = ax.plot(z, [func_midab(k, t0, l0, interval_0_1=True) for k in z], linewidth=1, color='purple')
# [line_cte1] = ax.plot(z, [1 for x in z], linewidth=1, color='red', linestyle='-')
[line_cte0] = ax.plot(z, [0 for x in z], linewidth=1, color='red', linestyle='-')
[mish_fixed] = ax.plot(z, [func_mish(k, interval_0_1=True) for k in z], linewidth=1, color='green', linestyle='dashed')
# [line_sigmoid] = ax.plot(z, [1/(1+np.exp(-x)) for x in z], linewidth=1, color='green', linestyle='dashed')
# [line_tanh] = ax.plot(z, np.tanh(z), linewidth=1, color='green', linestyle='dotted')

ax.set_xlim([min_x, max_x])
ax.set_ylim([min_y, max_y])
ax.grid(True)
ax.set_xlabel('x')
ax.set_ylabel('y')

# Add sliders for tweaking the parameters
tau_slider_ax = fig.add_axes([0.25, 0.15, 0.65, 0.03], facecolor=axis_color)
tau_slider = Slider(tau_slider_ax, 'tau', -15.0, 15.0, valinit=t0)
lambda_slider_ax = fig.add_axes([0.25, 0.1, 0.65, 0.03], facecolor=axis_color)
lambda_slider = Slider(lambda_slider_ax, 'lambda', -15.0, 15.0, valinit=l0)

# Define an action for modifying the line when any slider's value changes
def sliders_on_changed(val):
    # line_01.set_ydata([func_mila(k, tau_slider.val, interval_0_1=True) for k in z])
    line_02.set_ydata([func_midab(k, tau_slider.val, lambda_slider.val, interval_0_1=True) for k in z])
    fig.canvas.draw_idle()

lambda_slider.on_changed(sliders_on_changed)
tau_slider.on_changed(sliders_on_changed)

plt.show()