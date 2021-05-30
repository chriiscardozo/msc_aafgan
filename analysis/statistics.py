import argparse

from Utils import commons_utils
import numpy as np

### Definitions for statistic calculation
def print_top_accuracy(params):
    values = params['values']
    model_name = params['model_name']
    if len(values) == 0:
        max = -1
    else:
        max = values.mean(axis=0).max()
    print(model_name + ',' + "%.2f" % (max) + ',' + str(len(values)))


def top5_accuracy(params):
    values = params['values']
    model_name = params['model_name']
    if len(values) == 0:
        max = -1
    else:
        max = np.sort(values.mean(axis=0))[-5:].mean()
    print(model_name + ',' + "%.2f" % (max))


def epoch_to_percentace(params, percentage):
    values = params['values']
    model_name = params['model_name']
    if len(values) == 0:
        print(model_name + ',empty')
        return
    result = np.argwhere(values.mean(axis=0) >= percentage)
    if len(result) == 0:
        print(model_name + ',never')
        return
    print(model_name + ',' + str(result[0][0]))


def positive_deltas_percentage(params, start, end):
    values = params['values']
    model_name = params['model_name']
    baseline_stats = params['baseline_stats']
    if len(values) == 0:
        print(model_name + ',empty')
        return
    positive_deltas = ((values.mean(axis=0) > baseline_stats.mean(axis=0))[start:end].sum())/(end-start)*100
    print(model_name + ',' + "%.0f" % (positive_deltas))


def computational_failures(params):
    values = params['values']
    model_name = params['model_name']
    print(model_name + ',' + str(10-len(values)) + ' de 10')


def experiment_convergence_tax(params, threshold, fnt_compare, fnt_selection=lambda x: x.max(axis=1)):
    values = params['values']
    model_name = params['model_name']
    if len(values) == 0:
        print(model_name + ',empty')
        return

    qty = fnt_compare(fnt_selection(values), threshold).sum()
    print(model_name + ',' + str(qty) + ' de ' + str(len(values)))


def avg_epoch_time(params):
    values = params['values']
    model_name = params['model_name']

    if len(values) == 0:
        print(model_name + ',empty')
        return
    print(model_name, values[:][0:10].mean())


def print_top_fid(params):
    values = params['values']
    model_name = params['model_name']
    if len(values) == 0:
        min = 1000
    else:
        min = values.mean(axis=0).min()
    print(model_name + ',' + "%.2f" % (min) + ',' + str(len(values)))


def top5_fid(params):
    values = params['values']
    model_name = params['model_name']
    if len(values) == 0:
        min = 1000
    else:
        min = np.sort(values.mean(axis=0))[:5].mean()
    print(model_name + ',' + "%.2f" % (min))


def epoch_to_fid(params, score):
    values = params['values']
    model_name = params['model_name']

    if len(values) == 0:
        print(model_name + ',empty')
        return
    result = np.argwhere(values.mean(axis=0) <= score)
    if len(result) == 0:
        print(model_name + ',never')
        return
    print(model_name + ',' + str(result[0][0]))


def negative_deltas_fid(params, start, end):
    values = params['values']
    model_name = params['model_name']
    baseline_stats = params['baseline_stats']
    if len(values) == 0:
        print(model_name + ',empty')
        return
    negative_deltas = ((values.mean(axis=0) < baseline_stats.mean(axis=0))[start:end].sum())/(end-start)*100
    print(model_name + ',' + "%.0f" % (negative_deltas))
###


METRIC = {
    # FID score
    'top_fid': {
        'name': 'Top FID score',
        'filename': 'fid_scores.csv',
        'impl': lambda p: print_top_fid(p)
    },
    'top5_fid': {
        'name': 'Top-5 FID score médio',
        'filename': 'fid_scores.csv',
        'impl': lambda p: top5_fid(p)
    },
    'epochs_to_fid15': {
        'name': 'Épocas para atingir FID = 15',
        'filename': 'fid_scores.csv',
        'impl': lambda p: epoch_to_fid(p, 15)
    },
    'epochs_to_fid30': {
        'name': 'Épocas para atingir FID = 30',
        'filename': 'fid_scores.csv',
        'impl': lambda p: epoch_to_fid(p, 30)
    },
    'epochs_to_fid50': {
        'name': 'Épocas para atingir FID = 50',
        'filename': 'fid_scores.csv',
        'impl': lambda p: epoch_to_fid(p, 50)
    },
    # Accuracy
    'top_accuracy': {
        'name': 'Top Acurácia (%)',
        'filename': 'pre_classifier_accuracies.csv',
        'impl': lambda p: print_top_accuracy(p)
    },
    'top5_accuracy': {
        'name': 'Top-5 Acurácia (%)',
        'filename': 'pre_classifier_accuracies.csv',
        'impl': lambda p: top5_accuracy(p)
    },
    'epochs_to_50pct': {
        'name': 'Épocas para atingir acurácia 50%',
        'filename': 'pre_classifier_accuracies.csv',
        'impl': lambda p: epoch_to_percentace(p, 50)
    },
    'epochs_to_75pct': {
        'name': 'Épocas para atingir acurácia 75%',
        'filename': 'pre_classifier_accuracies.csv',
        'impl': lambda p: epoch_to_percentace(p, 75)
    },
    'epochs_to_90pct': {
        'name': 'Épocas para atingir acurácia 90%',
        'filename': 'pre_classifier_accuracies.csv',
        'impl': lambda p: epoch_to_percentace(p, 90)
    },
    # Deltas
    'pos_deltas_ep_0_50': {
        'name': 'Deltas positivos (época 1 à 50)',
        'filename': 'pre_classifier_accuracies.csv',
        'impl': lambda p: positive_deltas_percentage(p, 0, 50)
    },
    'pos_deltas_ep_50_100': {
        'name': 'Deltas positivos (época 51 à 100)',
        'filename': 'pre_classifier_accuracies.csv',
        'impl': lambda p: positive_deltas_percentage(p, 50, 100)
    },
    'pos_deltas_ep_0_100': {
        'name': 'Deltas positivos (época 1 à 100)',
        'filename': 'pre_classifier_accuracies.csv',
        'impl': lambda p: positive_deltas_percentage(p, 0, 100)
    },
    'fid_neg_deltas_ep_0_10': {
        'name': 'Deltas positivos (época 1 à 10)',
        'filename': 'fid_scores.csv',
        'impl': lambda p: negative_deltas_fid(p, 0, 10)
    },
    'fid_neg_deltas_ep_10_25': {
        'name': 'Deltas positivos (época 1 à 10)',
        'filename': 'fid_scores.csv',
        'impl': lambda p: negative_deltas_fid(p, 10, 25)
    },
    'fid_neg_deltas_ep_0_25': {
        'name': 'Deltas positivos (época 1 à 25)',
        'filename': 'fid_scores.csv',
        'impl': lambda p: negative_deltas_fid(p, 0, 25)
    },
    # falhas
    'accuracy_computational_failure': {
        'name': '% de falhas computationais',
        'filename': 'x.csv',
        'impl': lambda p: computational_failures(p)
    },
    'accuracy_experiment_convergence': {
        'name': '% de modelo convergindo',
        'filename': 'pre_classifier_accuracies.csv',
        'threshold_loadstat': 0,
        'impl': lambda p: experiment_convergence_tax(p, 90, (lambda a,b: a > b))
    },
    'fid_experiment_convergence': {
        'name': 'qtd de experimentos convergindo',
        'filename': 'fid_scores.csv',
        'threshold_loadstat': float('inf'),
        'impl': lambda p: experiment_convergence_tax(p, 25, (lambda a,b: a < b), (lambda x: x.min(axis=1)))
    },
    # tempo
    'avg_epoch_time': {
        'name': 'Tempo médio por época (s)',
        'filename': 'times.csv',
        'threshold_loadstat': 0,
        'impl': lambda p: avg_epoch_time(p)
    }
}


def get_arguments():
    parser = argparse.ArgumentParser(description='Calculate the statistics for accuracy metric')    
    parser.add_argument('--baseline', type=str, nargs=1, help='The baseline experiment directory', required=True)
    parser.add_argument('--experiments', type=str, nargs='+', help='The experiments directories to be compared', required=True)
    parser.add_argument('--metric', type=str, nargs='+', help='metric to analyse', choices=list(METRIC.keys()), required=True)
    parser.add_argument('--threshold', type=int, nargs=1, help='threshold to pass when loading stats')
    parser.add_argument('--threshold_lt', action='store_true', help='If we should use less-than comparison (instead of greater-than) for threshold comparison')
    parser.add_argument('--baseline_name', type=str, nargs=1, default=['Baseline'], help='the baseline name to be shown in graphs')
    args = parser.parse_args()
    return args


def order_condition(k):
    ORDER_LIST = ['BHSA', 'BHANA', 'BHATA', 'MiDA', 'Mish', 'SHReLU']
    index = [ORDER_LIST.index(x) for x in ORDER_LIST if x in k]
    index.append(0)
    return str(index[0]) + k


def generate_the_statistics(baseline_stats, experiments_folders, metric_id, baseline_name, threshold, threshold_lt):
    experiment_folder_name = experiments_folders[0].split('experiment_')[2].split('/')[0]
    
    print(experiment_folder_name, '/', METRIC[metric_id]['name'])

    experiments_folders = sorted(experiments_folders, key = lambda x: order_condition(x))

    for model_path in experiments_folders:
        if threshold is None:
            chosen_threshold = METRIC[metric_id]['threshold_loadstat'] if 'threshold_loadstat' in METRIC[metric_id] else 90
        else:
            chosen_threshold = threshold

        model_stats = commons_utils.load_stats(model_path, metric_file=METRIC[metric_id]['filename'], threshold=chosen_threshold, less_than=threshold_lt)
        try:
            model_name = model_path.split('output_')[1]
        except:
            # output
            model_name = 'baseline'

        METRIC[metric_id]['impl']({ 
                                    'values': model_stats, 
                                    'model_name': model_name, 
                                    'baseline_stats': baseline_stats
                                  })


def main():
    args = get_arguments()
    baseline_stats = commons_utils.load_stats(args.baseline[0], metric_file='fid_scores.csv')
    generate_the_statistics(baseline_stats, args.experiments, args.metric[0], args.baseline_name[0], args.threshold, args.threshold_lt)


if __name__ == '__main__':
    main()