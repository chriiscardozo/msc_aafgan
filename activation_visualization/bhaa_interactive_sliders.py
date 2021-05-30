from numpy import pi, sin
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import math

def func_bh_derivative_equal_to_zero(tau1, tau2, l):
    return [(tau1 - tau2)/(2*l*(tau1 + tau2)), (tau1 + tau2)/(2*l*(tau1 - tau2))]

def raw_func_bh_derivative(x, tau1, tau2, l):
    return (l * (1 - 2 * l * x))/math.sqrt(4 * tau2**2 + (1 - 2 * l * x)**2) + (l * (1 + 2 * l * x))/math.sqrt(4 * tau1**2 + (1 + 2 * l * x)**2)

def raw_func_bh(x, tau1, tau2, l):
    return (math.sqrt((l**2) * (x+(1/(2*l)))**2 + (tau1**2)) - math.sqrt((l**2) * (x-(1/(2*l)))**2 + (tau2**2)))

def func_bh(x, tau1, tau2, l, hard_limit, normalized, interval_0_1=False):
    if l > -0.001 and l < 0.001: l = 0.001
    result = raw_func_bh(x, tau1, tau2, l)
    if hard_limit and result > 1: result = 1
    if hard_limit and result <= -1: result = -1
    if normalized and tau1 != tau2:
        func_points_edge_value = func_bh_derivative_equal_to_zero(tau1, tau2, l)
        func_points_edge_value = max(func_points_edge_value) if abs(tau1) > abs(tau2) else min(func_points_edge_value)
        func_edge_value = raw_func_bh(func_points_edge_value, tau1, tau2, l)

        print(func_points_edge_value)

        if(abs(tau1) > abs(tau2)):
            result = (result - (-1))/(func_edge_value + 0.1 - (-1))
        elif(abs(tau2) > abs(tau1)):
            result = (result - func_edge_value)/(1 + 0.1 - func_edge_value)

        result = (result - 0.5) * 2

    if interval_0_1: result = (result/2) + 0.5
    return result

axis_color = 'lightgoldenrodyellow'

fig = plt.figure()
ax = fig.add_subplot(111)

# Adjust the subplots region to leave some space for the sliders and buttons
fig.subplots_adjust(left=0.25, bottom=0.25)

min_x = -15
max_x = 15
min_y = -3
max_y = 3
tau1_0 = .5
tau2_0 = .5
l_0 = .5
global hard_limit
global normalized
hard_limit = False
normalized = False

z = np.arange(min_x, max_x, .01)
y = [func_bh(k, tau1_0, tau2_0, l_0, hard_limit, normalized) for k in z]
y_linha = [raw_func_bh_derivative(k, tau1_0, tau2_0, l_0) for k in z]



# Draw the initial plot
# The 'line' variable is used for modifying the line later
[line] = ax.plot(z, y, linewidth=2, color='green')
[line_derivative] = ax.plot(z, y_linha, linewidth=1, color='blue')
[line_cte1] = ax.plot(z, [1 for x in z], linewidth=0.25, color='red')
[line_cte0] = ax.plot(z, [-1 for x in z], linewidth=0.25, color='red')
[line_sigmoid] = ax.plot(z, [1/(1+np.exp(-x)) for x in z], linewidth=0.5, color='purple')
[line_tanh] = ax.plot(z, np.tanh(z), linewidth=0.5, color='purple')

ax.set_xlim([min_x, max_x])
ax.set_ylim([min_y, max_y])
ax.grid(True)
ax.set_xlabel('x')
ax.set_ylabel('y')

# Add sliders for tweaking the parameters
tau1_slider_ax  = fig.add_axes([0.25, 0.15, 0.65, 0.03], facecolor=axis_color)
tau1_slider = Slider(tau1_slider_ax, 'tau1', -15.0, 15.0, valinit=tau1_0)

tau2_slider_ax = fig.add_axes([0.25, 0.1, 0.65, 0.03], facecolor=axis_color)
tau2_slider = Slider(tau2_slider_ax, 'tau2', -15.0, 15.0, valinit=tau2_0)

l_slider_ax = fig.add_axes([0.25, 0.05, 0.65, 0.03], facecolor=axis_color)
l_slider = Slider(l_slider_ax, 'lambda', -15.0, 15.0, valinit=l_0)

# Define an action for modifying the line when any slider's value changes
def sliders_on_changed(val):
    line.set_ydata([func_bh(k, tau1_slider.val, tau2_slider.val, l_slider.val, hard_limit, normalized) for k in z])
    line_derivative.set_ydata([raw_func_bh_derivative(k, tau1_slider.val, tau2_slider.val, l_slider.val) for k in z])
    fig.canvas.draw_idle()

tau1_slider.on_changed(sliders_on_changed)
tau2_slider.on_changed(sliders_on_changed)
l_slider.on_changed(sliders_on_changed)

# Add a set of radio buttons for changing color
hard_limit_radios_ax = fig.add_axes([0.025, 0.5, 0.15, 0.15], facecolor=axis_color)
hard_limit_radios = RadioButtons(hard_limit_radios_ax, ('no hard limit', 'hard limit', 'normalized'), active=0)
def hard_limit_radios_on_clicked(label):
    global hard_limit
    global normalized

    if label == 'no hard limit':
        hard_limit = False
        normalized = False
    elif label == 'hard limit':
        hard_limit = True
        normalized = False
    else:
        hard_limit = False
        normalized = True

    line.set_ydata([func_bh(k, tau1_slider.val, tau2_slider.val, l_slider.val, hard_limit, normalized) for k in z])
    line_derivative.set_ydata([raw_func_bh_derivative(k, tau1_slider.val, tau2_slider.val, l_slider.val) for k in z])

    fig.canvas.draw_idle()
hard_limit_radios.on_clicked(hard_limit_radios_on_clicked)

plt.show()