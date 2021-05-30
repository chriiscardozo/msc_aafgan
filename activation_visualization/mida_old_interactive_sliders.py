from numpy import pi, sin
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import math


def func_mish(x, interval_0_1=False):
    return x * np.tanh(np.log(1 + np.exp(x)))


def func_mila(x, b, interval_0_1=False):
    return x * np.tanh(np.log(1 + np.exp(x + b)))


def func_mida(x, a, b, interval_0_1=False):
    return a * x * np.tanh(np.log(1 + np.exp(x + b)))


axis_color = 'lightgoldenrodyellow'
fig = plt.figure()
ax = fig.add_subplot(111)

# Adjust the subplots region to leave some space for the sliders and buttons
fig.subplots_adjust(left=0.25, bottom=0.25)

min_x = -5
max_x = 5
min_y = -5
max_y = 5
a_0 = 1
b_0 = 1

z = np.arange(min_x, max_x, .01)

# Draw the initial plot
# The 'line' variable is used for modifying the line later
# [line_01] = ax.plot(z, [func_mila(k, b_0, interval_0_1=True) for k in z], linewidth=1, color='blue')
[line_02] = ax.plot(z, [func_mida(k, a_0, b_0, interval_0_1=True) for k in z], linewidth=1, color='purple')
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
beta_slider_ax = fig.add_axes([0.25, 0.15, 0.65, 0.03], facecolor=axis_color)
beta_slider = Slider(beta_slider_ax, 'beta', -15.0, 15.0, valinit=b_0)
alpha_slider_ax = fig.add_axes([0.25, 0.1, 0.65, 0.03], facecolor=axis_color)
alpha_slider = Slider(alpha_slider_ax, 'alpha', -15.0, 15.0, valinit=a_0)

# Define an action for modifying the line when any slider's value changes
def sliders_on_changed(val):
    # line_01.set_ydata([func_mila(k, beta_slider.val, interval_0_1=True) for k in z])
    line_02.set_ydata([func_mida(k, alpha_slider.val, beta_slider.val, interval_0_1=True) for k in z])
    fig.canvas.draw_idle()

alpha_slider.on_changed(sliders_on_changed)
beta_slider.on_changed(sliders_on_changed)

plt.show()