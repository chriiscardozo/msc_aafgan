import sys
# Add the ptdraft folder path to the sys.path list
sys.path.append('/home/ubuntu/msc_aafgan/')
from pre_trained_classifier.mnist_classifier import Net
import torch
from Utils import amazon_utils, dataset_utils, commons_utils, cuda_utils
import os
from sklearn.metrics import confusion_matrix
import csv
import ast
import numpy as np

import time

from multiprocessing.pool import ThreadPool
from multiprocessing import Lock

epochs_acc = {}
lock = Lock()

def load_pre_trained_model(dataset='MNIST'):
    if dataset == 'MNIST':
        model = Net().to(cuda_utils.DEVICE)
        model.load_state_dict(torch.load('pre_trained_classifier/mnist_cnn.pt', map_location=cuda_utils.DEVICE))
        model.eval()
    else:
        raise Exception("No pre trained classifier configured for database " + dataset)
    return model.to(cuda_utils.DEVICE)


def thread_safe_add_accuracy(epoch, value):
    global epochs_acc
    global lock
    lock.acquire()
    if epoch not in epochs_acc: epochs_acc[epoch] = []
    epochs_acc[epoch].append(value)
    lock.release()


def predict_from(X, model):
    X = X.view(-1, 1, 28, 28)
    predict = model(X)
    predict = predict.argmax(dim=1, keepdim=True)
    return predict


def calculate_accuracy(y_real, y_pred):
    sum_count = [0] * 10
    correct_count = [0] * 10
    
    for i in range(len(y_pred)):
        y_value = int(y_real[i].item())
        sum_count[y_value] += 1
        if y_value == y_pred[i]: correct_count[y_value] += 1

    hits = y_pred.eq(y_real.view_as(y_pred)).sum().item()
    acc = hits/len(y_real)*100.0
    return acc


def evaluate_single_epoch(arg_list):
    epoch = arg_list[0]
    path_to_seed = arg_list[1]
    model = arg_list[2]

    X = []
    Y = []
    filename = os.path.join(path_to_seed, commons_utils.DIR_SAMPLES_CSV, 'samples_'+str(epoch)+'.csv')
    with open(filename, 'r') as f:
        samples = f.readlines()

        x = map(lambda z: [float(a) for a in ast.literal_eval(z)[0][1:-1].split(',')], samples)
        y = map(lambda z: int(ast.literal_eval(z)[1].split('(')[1].split(',')[0].split(')')[0]), samples)
        X.extend(x)
        Y.extend(y)
    
    X = torch.FloatTensor(X).to(cuda_utils.DEVICE)
    Y = torch.Tensor(Y).to(cuda_utils.DEVICE)
    
    y_pred = predict_from(X, model)
    acc = calculate_accuracy(Y, y_pred)
    thread_safe_add_accuracy(epoch, acc)


def evaluate(model, base_path):
    for dir_model in os.listdir(base_path):
        if not os.path.exists(os.path.join(base_path, dir_model,'0',commons_utils.DIR_SAMPLES_CSV,'samples_0.csv')): continue

        if os.path.exists(os.path.join(base_path, dir_model, 'classifier_acc_avgs.csv')):
            print("Folder", dir_model, "done already, skipping...")
            continue

        print("Folder", dir_model)
        global epochs_acc
        epochs_acc = {}

        for seed in os.listdir(os.path.join(base_path, dir_model)):
            if not os.path.exists(os.path.join(base_path, dir_model,seed,commons_utils.DIR_SAMPLES_CSV,'samples_0.csv')): continue
            print("seed", seed)

            pool = ThreadPool(1) # TODO: change to number of cores in cpu # 287s

            start = time.time()

            params_list = range(len(os.listdir(os.path.join(base_path, dir_model, seed, commons_utils.DIR_SAMPLES_CSV))))
            params_list = [[x, os.path.join(base_path, dir_model, seed), model] for x in params_list]
            pool.map(evaluate_single_epoch, params_list)
            pool.close()
            pool.join()

            print("duration:", (time.time() - start)/60, "min")
        
        avgs = []
        stds = []
        for epoch in range(len(epochs_acc.keys())):
            avg = sum(epochs_acc[epoch])/len(epochs_acc[epoch])
            std = np.std(epochs_acc[epoch])
            avgs.append(avg)
            stds.append(std)

        with open(os.path.join(base_path, dir_model, 'classifier_acc_avgs.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(avgs)
        with open(os.path.join(base_path, dir_model, 'classifier_acc_stds.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(stds)


def main():
    if len(sys.argv) < 2:
        raise 'missing experiment folder name, ex: experiment_mnist_gan-1'

    BASE = os.path.join(amazon_utils.DIR_S3, sys.argv[1]) if amazon_utils.is_it_ec2() else sys.argv[1]
    model = load_pre_trained_model()
    print("Pre-trained model loaded...")

    result = evaluate(model, BASE)
    # with open(os.path.join(BASE, 'condgan_classifier_results.txt'), 'w') as f:
    #     f.write(str(result))
    amazon_utils.mark_spot_request_as_cancelled()
    amazon_utils.shutdown_if_ec2()
    exit(0)

    for dir_model in os.listdir(BASE):
        X = []
        Y = []

        current_output_dir = os.path.join(BASE, dir_model)

        for seed in os.listdir(current_output_dir):
            samples, labels = dataset_utils.get_manually_classified_data(os.path.join(current_output_dir, seed))
            X.extend(samples)
            Y.extend(labels)

        if(len(X) == 0): continue

        X = torch.FloatTensor(X)
        Y = torch.Tensor(Y)
        X = X.view(-1, 1, 28, 28)
        predict = model(X)
        predict = predict.argmax(dim=1, keepdim=True)
        sum_count = [0] * 10
        correct_count = [0] * 10

        for i in range(len(predict)):
            y_value = int(Y[i].item())
            sum_count[y_value] += 1
            if y_value == predict[i]: correct_count[y_value] += 1

        hits = predict.eq(Y.view_as(predict)).sum().item()

        print('Model:', dir_model)
        print("\tModel accuracy:", "{0:.2f}".format(hits/len(Y)*100.0))
        print("\tModel consufion matrix:")
        matrix = confusion_matrix(Y, predict)
        print(matrix)


if __name__ == "__main__":
    main()