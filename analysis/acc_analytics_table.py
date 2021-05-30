from Utils import commons_utils
import sys
import operator

import numpy as np
from Utils import commons_utils


def main():
    if len(sys.argv) < 2:
        print("Wrong usage. Command structure is:\n\tpython3 acc_analytics_file.py EXPERIMENT_FOLDER1/* [EXPERIMENT_FOLDER2/* ...]")
        exit(0)
    
    metrics = { "top1": {}, "top10_avg": {}, "first30_avg": {}, "last30_avg": {} }

    for model in sys.argv[1:]:
        accuracies = commons_utils.load_metric_evaluation.load_accuracies(model, threshold_convergence=50)
        metrics["top1"][model] = accuracies.mean(axis=0).max()
        metrics["first30_avg"][model] = accuracies[:,:30].mean()
        metrics["top10_avg"][model] = np.sort(accuracies.mean(axis=0))[-10:].mean()
        metrics["last30_avg"][model] = accuracies[:,-30:].mean()

    for k in metrics.keys():
        print(k)
        ranked = sorted(metrics[k].items(), key=operator.itemgetter(1))
        ranked.reverse()
        for name, value in ranked: print("%s: %.4f" % (name, value))


if __name__ == '__main__':
    main()