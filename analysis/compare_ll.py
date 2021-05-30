import numpy as np
import itertools
import sys, os, csv
import matplotlib.pyplot as plt
from Utils import commons_utils

DIR_LLS_COMPARE='lls_comparisons'
FINAL_DIR=None

def compare(folders=[],multi_seeds=True):
    global FINAL_DIR
    # if(len(sys.argv) < 2):
    #     print("Missing result folders parameters")
    #     exit(0)
    if(len(folders) == 0):
        folders = commons_utils.find_output_folders()

    x = []
    x_folder_sample = os.path.join(folders[0], commons_utils.DIR_STATS, 'x.csv')
    if(multi_seeds): x_folder_sample = os.path.join(folders[0], '0', commons_utils.DIR_STATS, 'x.csv')
    with open(x_folder_sample) as f:
        reader = csv.reader(f)
        x = [int(k) for k in next(reader)]

    start=0
    stop=None
    step=1
    x = x[start:stop:step]
    FINAL_DIR = DIR_LLS_COMPARE+"_"+str(start)+"_"+str(stop)+"_"+str(step)
    commons_utils.reset_dir(FINAL_DIR)

    combinations = itertools.combinations(folders, 2)

    for e in combinations:
        output_name = (e[0]+"_vs_"+e[1]).replace('/','_')
        compare_lls(x,e,output_name,start,stop,step,multi_seeds)
    
    compare_lls(x,folders, "all",start,stop,step,multi_seeds)
    

def compare_lls(x,folders,output_name,start,stop,step,multi_seeds):

    plt.clf()
    plt.title("GAN MNIST\nNegative Log-Likelihood per epoch (log-scalar scale)")
    plt.ylabel('Negative Log-Likelihood')
    plt.xlabel('epoch')
    #plt.yscale('log')

    legends = []

    for folder in folders:
        print("Doing for:", folder)
        if(multi_seeds):
            lls_avg = []
            for e in os.listdir(folder):
                print("seed", e)
                if not os.path.exists(os.path.join(folder,e,commons_utils.DIR_STATS,'lls_avg.csv')): continue
                f = open(os.path.join(folder,e,commons_utils.DIR_STATS,'lls_avg.csv'))
                reader = csv.reader(f)
                values = [-float(x) for x in next(reader)]
                values = [x if x < 10000 else 10000 for x in values]
                lls_avg.append(values)
            lls_avg = np.mean(lls_avg, axis=0).tolist()
        else:
            f = open(os.path.join(folder, commons_utils.DIR_STATS, 'lls_avg.csv'))
            reader = csv.reader(f)
            lls_avg = [-float(x) for x in next(reader)]
        
        lls_avg = lls_avg[start:stop:step]
        line, = plt.plot(x, lls_avg, label=folder.replace('/','_'))
        legends.append(line)

    plt.legend(handles=legends)
    plt.savefig(os.path.join(FINAL_DIR,output_name+'.png'))
    # plt.show()

def main():
    folders = []
    if(len(sys.argv) > 1):
        for f in sys.argv[1:]:
            if("output" in f):
                folders.append(f)

    compare(folders)

if __name__ == '__main__':
    main()