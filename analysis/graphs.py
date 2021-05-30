import argparse

from Utils import commons_utils
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np

plt.style.use('ggplot')

METRIC = {
    'accuracy': {
        'name': 'Acurácia (%)',
        'filename': 'pre_classifier_accuracies.csv'
    },
    'fid': {
        'name': 'FID score',
        'filename': 'fid_scores.csv'
    }
}


def get_arguments():
    parser = argparse.ArgumentParser(description='Build the graphs for experiment metrics')
    parser.add_argument('--baseline', type=str, nargs=1, help='The baseline experiment directory', required=True)
    parser.add_argument('--experiments', type=str, nargs='+', help='The experiments directories to be compared', required=True)
    parser.add_argument('--metric', type=str, nargs='+', help='metric to analyse', choices=['accuracy', 'fid'], required=True)
    parser.add_argument('--split_graph', nargs='+', type=int, default=[], help='split the graph visualization into provided list of epochs')
    parser.add_argument('--together', type=str, default='', help='if the graphs should be rendered all together with provided name')
    parser.add_argument('--baseline_name', type=str, nargs=1, default=['Baseline'], help='the baseline name to be shown in graphs')
    parser.add_argument('--diff', action='store_true', help='if graphs should be difference between baseline and models')
    parser.add_argument('--figsize', type=float, nargs=2, default=[5,5], help='the output figsize')
    args = parser.parse_args()
    return args


def make_plot(x, values, name, ax):
    x_new, power_smooth = make_smooth_data(x, values)
    line, = ax.plot(x_new, power_smooth, label=name, lw=1)
    return line


def make_smooth_data(x, values):
    x_new = np.linspace(x.min(), x.max(), len(x)*2)
    spl = make_interp_spline(x, values, k=3)
    return x_new, spl(x_new)


def make_fill_plot(x, lower, upper, ax):
    _, min_smooth = make_smooth_data(x, lower)
    x_new, max_smooth = make_smooth_data(x, upper)
    ax.fill_between(x_new, min_smooth, max_smooth, alpha = 0.15)


def get_min_max_avg_from_range(values, start, end):
    return values.min(axis=0)[start:end], values.max(axis=0)[start:end], values.mean(axis=0)[start:end]


def save_the_fig(ax, fig, subplot_adjustmet, output_path):
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', prop={'size': 9})
    fig.tight_layout()
    plt.subplots_adjust(bottom=subplot_adjustmet)
    plt.savefig(output_path)
    

def generate_graphs(baseline_stats, experiments_folders, metric_id, split_points, is_together, together_name, 
                    baseline_name, is_diff, figsize):

    experiment_folder_name = experiments_folders[0].split('experiment_')[2].split('/')[0]
    split_points.append(len(baseline_stats[0]))
    fig = ax = None

    for model_path in experiments_folders:
        model_stats = commons_utils.load_stats(model_path, metric_file=METRIC[metric_id]['filename'])
        model_name = model_path.split('output_')[1]

        if not len(model_stats) > 0: continue
        print(model_name,':', len(model_stats))

        # fig, ax
        if fig is None or not is_together:
            fig, ax = plt.subplots(nrows=min(2, len(split_points)), ncols=max(1, int(len(split_points)/2)), figsize=figsize)
            title_model_name = together_name if is_together else model_name
            # fig.suptitle(('(Delta) ' if is_diff else '') + baseline_name + ' vs ' + title_model_name, fontsize=12.5)

        ax = np.array(ax).flatten()

        start_point = 0
        for i in range(len(split_points)):
            end_point = split_points[i]

            baseline_min, baseline_max, baseline_avg = get_min_max_avg_from_range(baseline_stats, start_point, end_point)
            model_min, model_max, model_avg = get_min_max_avg_from_range(model_stats, start_point, end_point)

            x = np.array(range(len(baseline_stats[0])))[start_point:end_point]

            if is_diff:
                text_model = 'modelo' if not is_together else model_name
                make_plot(x, model_avg-baseline_avg, 'Delta = ' + text_model + ' - ' + baseline_name, ax[i])
                ax[i].plot(x, [0 for _ in x], linewidth=0.75, ls='dashed')
            else:
                make_fill_plot(x, model_min, model_max, ax[i])
                make_fill_plot(x, baseline_min, baseline_max, ax[i])

                make_plot(x, model_avg, model_name, ax[i])
                make_plot(x, baseline_avg, baseline_name, ax[i])

            # axis labels
            ax[i].set_xlabel('Época', fontsize=10)
            ax[i].set_ylabel(('Delta ' if is_diff else '') + METRIC[metric_id]['name'], fontsize=10)
            ax[i].tick_params(axis='both', which='major', labelsize=9)
            # axis limits
            ax[i].set_xlim([start_point, end_point-1])
            
            # misc axis configurations
            ax[i].xaxis.get_major_locator().set_params(integer=True)
            ax[i].grid(True)
            start_point = end_point
        
        if not is_together:
            save_the_fig(ax, fig, (0.4 if is_diff else 0.3), 
                        'out/'+experiment_folder_name+'_'+baseline_name+'_vs_'+model_name+('_diff' if is_diff else '') +'.pdf')
    
    if is_together:
        save_the_fig(ax, fig, (0.25 if is_diff else 0.3),
                    'out/'+experiment_folder_name+'_'+baseline_name+'_together_'+together_name + ('_diff' if is_diff else '') +'.pdf')

def main():
    args = get_arguments()
    baseline_stats = commons_utils.load_stats(args.baseline[0], metric_file=METRIC[args.metric[0]]['filename'])

    is_together = len(args.together) > 0
    together_name = args.together if is_together else None

    generate_graphs(baseline_stats, args.experiments, args.metric[0], args.split_graph, is_together, together_name, args.baseline_name[0], args.diff,
                    args.figsize)
    

if __name__ == '__main__':
    main()