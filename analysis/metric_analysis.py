import sys, os
import numpy as np
from numpy.lib.function_base import diff
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from Utils import commons_utils

plt.style.use('ggplot')

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


def make_plot(values, name, ax, MA=False):
    if MA: values = [moving_average(values, z, 2) for z in range(len(values))]
    x = np.array(range(len(values)))
    x_new = np.linspace(x.min(), x.max(), len(x)*5)
    spl = make_interp_spline(x, values, k=5)
    power_smooth = spl(x_new)
    line, = ax.plot(x_new, power_smooth, label=name)
    return line


def make_textbox(ax, strtext, pos_x=0.8, posy=0.6):
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(pos_x, posy, strtext, transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='left', bbox=props)


def main():
    if len(sys.argv) < 3:
        print("Wrong usage. Command structure is:\n\tpython3 metric_analysis.py ACTION DEFAULT_OUTPUT_FOLDER_MODEL [EXPERIMENT_FOLDERS...]")
        exit(0)
    
    is_diff = sys.argv[1] == 'diff'
    default_dir = sys.argv[2]
    model_dirs = sys.argv[3:]
    
    default_accuracies = commons_utils.load_metric_evaluation(default_dir, 90)
    default_accuracies = default_accuracies.mean(axis=0)

    for model_path in model_dirs:
        model_name = model_path.split('/')[-1]
        print("{} in path: {}".format(model_name, model_path))

        fig, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(8,6))

        model_accuracies = commons_utils.load_metric_evaluation(model_path, 90)
        if(len(model_accuracies) == 0): continue
        model_accuracies = model_accuracies.mean(axis=0)

        if is_diff:
            difference = model_accuracies - default_accuracies
            title = 'Diferença entre acurácias: delta = (modelo - baseline)'
            ylabel = 'Delta acurácia (%)'
        else: 
            difference = model_accuracies
            title = ''
            ylabel = 'Acurácia (%)'
        
        ax0.set_xlim([0,100])
        more_ticks(ax0)

        make_plot(difference, model_name, ax0, MA=False)
        ax0.set_title(title)
        ax0.set_xlabel('Época')
        ax0.set_ylabel(ylabel)
        ax0.plot(range(len(model_accuracies)), [0 for _ in model_accuracies], linewidth=1, ls='dashed',label='baseline')
        
        if is_diff:
            x_max = np.argmax(difference)
            y_max = max(difference)
            x_min = np.argmin(difference)
            y_min = min(difference)

            strtext = "Melhor delta: ep={:d}, v={:.2f}%".format(x_max, y_max)
            strtext += "\nPior delta: ep={:d}, v={:.2f}%".format(x_min, y_min)
            strtext += "\nTotal deltas positivos: {:d}".format(len([x for x in difference if x >= 0]))
            strtext += "\nTotal deltas negativos: {:d}".format(len([x for x in difference if x < 0]))
            strtext += "\n% de delta > 0: épocas [1,50]: {:.0f}%".format(len([x for x in difference[:50] if x >= 0])/50*100)
            strtext += "\n% de delta > 0: épocas [51,100]: {:.0f}%".format(len([x for x in difference[50:] if x >= 0])/50*100)
            # make_textbox(ax0, strtext)
            plt.plot([], [], ' ', label=strtext)

        ax0.grid(True)
        ax0.legend()
        plt.savefig('out/metric_analytics'+model_path.replace('/', '_')+'.png')


if __name__ == '__main__':
    main()