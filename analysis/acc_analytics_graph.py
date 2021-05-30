import sys, os
import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from Utils import commons_utils


def more_ticks(ax):
    loc_x = plticker.MultipleLocator(base=5.0)
    loc_y = plticker.MultipleLocator(base=2.0)
    ax.xaxis.set_major_locator(loc_x)
    ax.yaxis.set_major_locator(loc_y)


def moving_average(values, index, k):
    data = []
    data.append(values[index])
    for i in range(k):
        if index - i >= 0: data.append(values[index-i])
        else: break
    return sum(data)/len(data)


def make_plot(values, name, ax, MA=True):
    if MA:
        values = [moving_average(values, z, 2) for z in range(len(values))]
    x = np.array(range(len(values)))
    x_new = np.linspace(x.min(), x.max(), len(x)*5)
    spl = make_interp_spline(x, values, k=5)
    power_smooth = spl(x_new)
    line, = ax.plot(x_new, power_smooth, label=name)
    return line


def make_textbox(ax, strtext, pos_x=0.05, posy=0.15):
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(pos_x, posy, strtext, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)


def main():
    if len(sys.argv) < 3:
        print("Wrong usage. Command structure is:\n\tpython3 acc_analytics_file.py DEFAULT_OUTPUT_FOLDER_MODEL EXPERIMENT_FOLDERS")
        exit(0)
    
    default_dir = sys.argv[1]
    experiment_dir = sys.argv[2]
    
    default_accuracies = commons_utils.load_accuracies_avg(default_dir)

    for model in os.listdir(experiment_dir):
        print("model", model)

        fig, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
        
        model_accuracies = commons_utils.load_accuracies_avg(os.path.join(experiment_dir, model))
        try:
            model_accuracies = commons_utils.load_accuracies_avg(os.path.join(experiment_dir, model))
        except:
            continue

        difference = model_accuracies - default_accuracies


        make_plot(difference, 'difference', ax0, MA=False)
        ax0.set_title('Accuracy diff: default output vs ' + os.path.join(experiment_dir,  model))
        ax0.plot(range(len(model_accuracies)), [0 for _ in model_accuracies], linewidth=1, color='red', linestyle='-')
        x_max = np.argmax(difference)
        y_max = max(difference)
        x_min = np.argmin(difference)
        y_min = min(difference)

        strtext = "Better: e={:d}, v={:.2f}%".format(x_max, y_max)
        strtext += "\nWorst: e={:d}, v={:.2f}%".format(x_min, y_min)
        strtext += "\nPositive datapoints: {:d}".format(len([x for x in difference if x >= 0]))
        strtext += "\nNegative datapoints: {:d}".format(len([x for x in difference if x < 0]))
        strtext += "\nPositive dp % (0:50): {:.0f}%".format(len([x for x in difference[:50] if x >= 0])/50*100)
        strtext += "\nPositive dp % (50:100): {:.0f}%".format(len([x for x in difference[50:] if x >= 0])/50*100)
        make_textbox(ax0, strtext)

        plt.show()


if __name__ == '__main__':
    main()