import os
from pickle import load

from Utils import commons_utils
import ast
import numpy as np

from multiprocessing import Pool

# This script will convert the csv samples in experiment folders to npz files.
# The design on how to save the epochs samples has changed to be a folder naming the epoch with several npz samples files inside it.
# Old experiments had csv of samples. After running this, we will have only the samples as compressed npz.
#   Old structure:
#       samples_csv/[samples_0.csv,samples_1.csv,...]
#   New structure (samples_csv/epoch/sample_batchrange):
#       samples_csv/[0,1,2,...]/[samples_0_1000.npz,samples_1000_2000.npz,...]
# For simplicity, the folder name "samples_csv" has not been changed.

def doit(experiment, model, seed):
    for sample in os.listdir(os.path.join(experiment, model, seed, commons_utils.DIR_SAMPLES_CSV)):
        if 'samples_' not in sample: continue
        filename = os.path.join(experiment, model, seed, commons_utils.DIR_SAMPLES_CSV, sample)

        epoch = sample.split('_')[1].split('.')[0]

        with open(filename, 'r') as f:
            samples = f.readlines()

            x = np.array(list(map(lambda z: [float(a) for a in ast.literal_eval(z)[0][1:-1].split(',')], samples)))
            y = np.array(list(map(lambda z: int(ast.literal_eval(z)[1].split('(')[1].split(',')[0].split(')')[0]), samples)))

            commons_utils.reset_dir(os.path.join(experiment, model, seed, commons_utils.DIR_SAMPLES_CSV, epoch))

            save_in = os.path.join(experiment, model, seed, commons_utils.DIR_SAMPLES_CSV, epoch, 'samples_' + str(0) + '_' + str(len(x)) + '.npz')

            np.savez_compressed(save_in, generatedImages=x, noise_in_labels=y)

            loaded_dict = np.load(save_in)
            if not np.array_equal(loaded_dict['noise_in_labels'], y) or not np.array_equal(loaded_dict['generatedImages'], x):
                raise Exception("Arrays comparison must match: " + str(np.array_equal(loaded_dict['noise_in_labels'], y)) + " " + str(np.array_equal(loaded_dict['generatedImages'], x[0])))

            os.remove(filename)


def main():
    for experiment in os.listdir('.'):
        if 'experiment_' not in experiment or 'cifar10' in experiment: continue
        print('experiment', experiment)

        pool = Pool(processes=10)

        for model in os.listdir(experiment):
            if not os.path.isdir(os.path.join(experiment, model)): continue

            print('model', model)
            for seed in os.listdir(os.path.join(experiment, model)):
                if not os.path.exists(os.path.join(experiment, model, seed, commons_utils.DIR_SAMPLES_CSV, 'samples_0.csv')): continue
                print('seed', seed)
                pool.apply_async(doit, (experiment, model, seed))
        pool.close()
        pool.join()


if __name__ == "__main__":
    main()






    