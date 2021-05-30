import os
import csv
import numpy as np

DATA_DIRS = {
    'dcgan_aws': '~/msc_times/time_evaluation_experiment_celeba_dcgan_aws',
    'dcgan_galeao':  '~/msc_times/time_evaluation_experiment_celeba_dcgan_galeao',
    'condgan_aws':  '~/msc_times/time_evaluation_experiment_condgan_mlp_aws',
    'condgan_galeao':  '~/msc_times/time_evaluation_experiment_condgan_mlp_galeao',
}


for key in DATA_DIRS.keys():
    print("--->", key)
    experiment_folder = DATA_DIRS[key]

    for model in os.listdir(experiment_folder):
        times = []
        for seed in os.listdir(os.path.join(experiment_folder, model)):
            if seed == 'config.json': continue
            with open(os.path.join(experiment_folder, model, seed, 'stats', 'times.csv')) as f:
                csvreader = csv.reader(f)
                values = [float(x) for x in f.readlines()[0].split(',')]
                times.extend(values)
        times = np.array(times)

        print("%s: %.2f, %.2f, %.2f" % (model, times.mean(), times.mean()-times.min(), times.mean()-times.max()))