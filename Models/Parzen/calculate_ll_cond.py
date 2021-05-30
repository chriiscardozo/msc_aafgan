import Parzen as P
import os
import sys
import csv
import numpy as np
from Utils import dataset_utils, commons_utils
import torch
import ast
import time

def do_parzen(folders, multi_seeds=True):
    test_data = dataset_utils.get_mnist_data(train=False)
    X_test = torch.utils.data.DataLoader(test_data, batch_size=10000, shuffle=True, pin_memory=1, num_workers=commons_utils.cpu_count())
    Y_test = next(iter(X_test))[1]
    X_test = next(iter(X_test))[0].view(10000, -1)

    classes_codes = torch.unique(Y_test, sorted=True)

    if(len(folders) == 0):
        folders = commons_utils.find_output_folders()
    
    for folder in folders:
        if not multi_seeds:
            raise Exception("old structure is not expected to exist anymore")
        else:
            for s in sorted(os.listdir(folder), key = lambda x: int(x) if len(x) < 3 else -1):
                if not os.path.exists(os.path.join(folder,s,commons_utils.DIR_SAMPLES_CSV,'samples_0.csv')): continue

                lls_std = []
                lls_avg = []

                for e_file in sorted(os.listdir(os.path.join(folder,s,commons_utils.DIR_SAMPLES_CSV)), key = lambda x: int(x.split('_')[1].split('.')[0])):
                    file_path = os.path.join(os.path.join(folder,s,commons_utils.DIR_SAMPLES_CSV), e_file)

                    start = time.time()
                    print("Doing", file_path)

                    with open(file_path, 'r') as f:
                        reader = csv.reader(f, delimiter=',')
                        samples = []
                        for row in reader: samples.append(row)
                        samples = np.array(samples)

                        if(len(samples[0]) == 2): # conditional model
                            samples_classes = np.array([int(x[1].split('(')[1].split(',')[0]) for x in samples])
                            samples = np.array([ast.literal_eval(x[0]) for x in samples])
                        else:
                            # non-conditional model
                            raise Exception("Pending impl")

                        classes_lls_avgs = []
                        classes_lls_stds = []
                        for class_code in classes_codes:
                            # debug like pry
                            # import code; code.interact(local=dict(globals(), **locals()))

                            class_code = class_code.item()
                            current_X_test = X_test[Y_test == class_code]
                            current_samples = samples[samples_classes == class_code]

                            ll_avg, ll_std = P.log_prob(current_X_test, torch.tensor(current_samples), gpu=False, class_code=class_code)
                            classes_lls_avgs.append(ll_avg)
                            classes_lls_stds.append(ll_std)
                        lls_avg.append(np.mean(classes_lls_avgs))
                        lls_std.append(np.mean(classes_lls_stds))
                    
                    duration = time.time() - start
                    print("%.2f s" % duration)
                
                output_dir=os.path.join(folder,s)
                commons_utils.save_general_information( {"lls_avg_conditional": lls_avg, "lls_std_conditional": lls_std}, 
                                                        output_dir)
                commons_utils.generate_graphics_lls(range(len(lls_avg)),lls_avg,output_dir,cond=True)

def main():
    folders = []
    if(len(sys.argv) > 1):
        for f in sys.argv[1:]:
            if("output" in f):
                folders.append(f)

    do_parzen(folders)

if __name__ == "__main__":
    main()