import multiprocessing
import os, shutil, csv, json
import time
from Utils import cuda_utils
import numpy as np
import random
import matplotlib.pyplot as plt
from torchsummary import summary
import io
from contextlib import redirect_stdout
from pre_trained_classifier import classifier_accuracy
from torchvision.utils import save_image
import torch

plt.switch_backend('agg')

# ------------
SUMMARY_FILE = 'summary.txt'
DIR_SAMPLES_CSV = 'samples_csv'
DIR_SAMPLES_IMG = 'samples_img'
DIR_STATS = 'stats'
DIR_GRAPHICS = 'graphics'
FILE_MARK_MODEL_COMPLETED='config.json'
FILE_MARK_MODEL_ERROR='cuda_error.json'
FILE_MARK_MODEL_DOUBLE_ERROR='cuda_double_error.json'
# ------------

def find_output_folders(folder='.', prefix='output',sorted_fnt=lambda x: x):
    filenames = os.listdir(folder)
    return sorted([ filename for filename in filenames if filename.startswith( prefix ) ], key=sorted_fnt)

def reset_dir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

def cpu_count():
    return multiprocessing.cpu_count()

def get_time():
    return time.time()

def exec_time(start, msg):
    end = time.time()
    delta = end - start
    if(delta > 60): print("Exec time: " + str(int(delta/60.0)) + " min [" + msg + "]")
    else: print("Exec time: " + str(int(delta)) + " s [" + msg + "]")

def build_output_dir_path(config, d_model, g_model, d_hidden_model, g_hidden_model):
    output_dir = "output"
    if(d_hidden_model != "default"): output_dir += "_D" + d_hidden_model
    if(g_hidden_model != "default"): output_dir += "_G" + g_hidden_model
    if(d_model != "default"): output_dir += "_Dis_" + d_model
    if(g_model != "default"): output_dir += "_Gen_" + g_model
    output_dir = os.path.join(config["EXPERIMENT_DIR"], output_dir)
    return output_dir

# def get_model_by_dir_name(name):
#     d_model = 'default'
#     g_model = 'default'
#     d_shrelu = 1 if 'Dshrelu' in name else 0
#     g_shrelu = 1 if 'Gshrelu' in name else 0
#     if 'Dis_' in name:
#         name.split('Dis_')

def model_completed(output_dir):
    return os.path.exists(output_dir) and os.path.exists(os.path.join(output_dir, FILE_MARK_MODEL_COMPLETED))

def model_marked_as_error(output_dir):
    return os.path.exists(output_dir) and os.path.exists(os.path.join(output_dir, FILE_MARK_MODEL_ERROR))

def model_marked_as_double_error(output_dir):
    return os.path.exists(output_dir) and os.path.exists(os.path.join(output_dir, FILE_MARK_MODEL_DOUBLE_ERROR))

def set_seed_as(value):
    cuda_utils.set_seed_as(value)
    random.seed(value)
    np.random.seed(value)

def mark_model_as_completed(output_dir, config):
    with open(os.path.join(output_dir, FILE_MARK_MODEL_COMPLETED), 'w') as f:
        json.dump(config, f)

def mark_model_as_cuda_error(output_dir, exception):
    with open(os.path.join(output_dir, FILE_MARK_MODEL_ERROR), 'w') as f:
        json.dump({"error": str(exception)}, f)

def mark_model_as_double_cuda_error(output_dir, exception):
    with open(os.path.join(output_dir, FILE_MARK_MODEL_DOUBLE_ERROR), 'w') as f:
        json.dump({"error": str(exception)}, f)

def generate_visualization_samples(epoch, visualization_noise, visualization_noise_labels, G, output_dir, channels, vector_dim=28, dim=(5, 5), figsize=(5, 5)):
    generatedImages = cuda_utils.vectors_to_images(G(visualization_noise, visualization_noise_labels).detach(), vector_dim, channels).data
    generatedImages = generatedImages.reshape(generatedImages.size()[0], channels, vector_dim, vector_dim).cpu()

    if(generatedImages.size()[0] > 25):
        if generatedImages.size()[0] % 5 != 0: raise Exception("SAMPLES_IN_VISUALIZATION should divide by 5")
        dim=(5,generatedImages.size()[0]/5)
        figsize=dim

    if channels == 1: # MNIST
        generatedImages = generatedImages.reshape(generatedImages.size()[0], vector_dim, vector_dim).cpu()

        # saving as img
        plt.close('all')
        plt.figure(figsize=figsize)
        for i in range(generatedImages.shape[0]):
            plt.subplot(dim[0], dim[1], i+1)
            plt.imshow(generatedImages[i], interpolation='nearest', cmap='gray_r')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, DIR_SAMPLES_IMG, str(epoch) + '.png'))
    elif channels == 3: # CIFAR10
        generatedImages = generatedImages.reshape(generatedImages.size()[0], channels, vector_dim, vector_dim).cpu()
        save_image(generatedImages, os.path.join(output_dir, DIR_SAMPLES_IMG, str(epoch) + '.png'), nrow=5, normalize=True)
    else:
        raise Exception("Not expected channels = " + str(channels))

def generate_epoch_samples(config, epoch, G, output_dir, noise_dim, channels, is_conditional=False, n_csv=1000, vector_dim=28, is_conv=False, N_CSV_LIMIT=1000):
    return_images = n_csv <= N_CSV_LIMIT
    noise_in = cuda_utils.noise(n_csv, noise_dim, channels)
    noise_in_labels = None
    if is_conditional: noise_in_labels = cuda_utils.fake_labels(n_csv, config["DOUBLE_TENSORS"])
    if(is_conv): noise_in = noise_in.view(-1, 100, 1, 1)

    # we genarate in iteratively in batches when it is more than 1k because it might not fit in GPU memory
    # the batches will be inside a folder which name is the epoch
    # MNIST experiments we generate 1K; CelebA experiments we generate 10K
    if True:#n_csv > N_CSV_LIMIT:
        print("n_csv ", n_csv, "is greater than", N_CSV_LIMIT, ", so saving generatedImages for epoch in batches using npz compressed mode")
        arr_length = n_csv

        epoch_samples_dir = os.path.join(output_dir, DIR_SAMPLES_CSV, str(epoch))
        reset_dir(epoch_samples_dir)

        allGeneratedImages = torch.Tensor()
        while n_csv > 0:
            start_index = arr_length - n_csv
            end_index = (arr_length - n_csv) + N_CSV_LIMIT
            if end_index > arr_length: end_index = arr_length

            noise_in_aux = noise_in[start_index:end_index]
            noise_in_labels_aux = noise_in_labels[start_index:end_index] if noise_in_labels is not None else None
            generatedImages = G(noise_in_aux, noise_in_labels_aux).detach().data.to(torch.device('cpu'))
            allGeneratedImages = torch.cat((allGeneratedImages, generatedImages), dim=0)

            n_csv -= N_CSV_LIMIT

        filepath = os.path.join(epoch_samples_dir, 'samples_' + str(0) + '_' + str(arr_length) + '.npz')
        np.savez_compressed(filepath, generatedImages=allGeneratedImages)
        if return_images:
            generatedImages = allGeneratedImages.to(cuda_utils.DEVICE)
        else:
            generatedImages = noise_in_labels = None # set to None as flag so caller knows it saved in batch

    return generatedImages, noise_in_labels

def save_general_information(values_dict, output_dir):
    for k in values_dict:
        value = values_dict[k]
        if(len(value) == 0): continue
        with open(os.path.join(output_dir, DIR_STATS, k + ".csv"), "w") as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(value)

def generate_graphics(x, d_lossses, g_losses, d_accuracies, lls_avg, lls_std, activation_parameters, output_dir):
    plt.close('all')

    plt.clf()
    plt.title("GAN MNIST - D and G losses per epoch")
    plt.ylabel('loss(binary crossentropy)')
    plt.xlabel('epoch')
    plt.plot(x, d_lossses, 'b-', label="D loss")
    plt.plot(x, g_losses, 'g-', label="G loss")
    plt.savefig(os.path.join(output_dir, DIR_GRAPHICS, 'losses.png'))
    # plt.show()

    plt.clf()
    plt.title("GAN MNIST - Disciminator accuracy per epoch")
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.plot(x, d_accuracies)
    plt.savefig(os.path.join(output_dir, DIR_GRAPHICS, 'accuracy.png'))
    # plt.show()

    if lls_avg is not None and len(lls_avg) > 0:
        generate_graphics_lls(x,lls_avg,output_dir)

    for model in activation_parameters.keys():
        for metric in activation_parameters[model].keys():
            if(len(activation_parameters[model][metric]) > 0):
                generate_graphics_activation_parameters(x, [aux[0] for aux in activation_parameters[model][metric]], 
                                               [aux[1] for aux in activation_parameters[model][metric]],
                                               [aux[2] for aux in activation_parameters[model][metric]],
                                               "GAN MNIST - " + model + " " + metric + " per epoch", "epoch", metric + " (max/avg/min)",
                                               output_dir, model + "_" + metric)

def generate_graphics_lls(x,lls_avg,output_dir,cond=False):
    plt.clf()
    plt.title("GAN MNIST - Negative Log-Likelihood per epoch (log-scalar scale)")
    plt.ylabel('Negative Log-Likelihood')
    plt.xlabel('epoch')

    prefix = '' if not cond else 'cond_'

    if(abs(max(lls_avg)-min(lls_avg)) > 1000):
        plt.yscale('log')
    plt.text(max(x)/2,-min(lls_avg)/10,'min = ' + str(-int(max(lls_avg))) + '\nepoch = ' + str(x[np.array(lls_avg).argmax()]))
    plt.plot(x, [-float(x) for x in lls_avg])
    plt.savefig(os.path.join(output_dir, DIR_GRAPHICS, prefix+'ll.png'))
    # plt.show()

# Reference for the below function: https://www.datascience.com/learn-data-science/tutorials/creating-data-visualizations-matplotlib-data-science-python
def generate_graphics_activation_parameters(x, y_data, upper_CI, low_CI, title, x_label, y_label, output_dir, filename):
    plt.clf()
    # Create the plot object
    _, ax = plt.subplots()    

    # Plot the data, set the linewidth, color and transparency of the
    # line, provide a label for the legend
    ax.plot(x, y_data, lw = 1, color = '#539caf', alpha = 1, label = 'Avg')
    # Shade the confidence interval
    ax.fill_between(x, low_CI, upper_CI, color = '#539caf', alpha = 0.4, label = '[Min-Max]')
    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Display legend
    ax.legend(loc = 'best')

    plt.text(0,min(low_CI),'final min = ' + str(low_CI[-1]) + '\nfinal max = ' + str(upper_CI[-1]) + '\nfinal med = ' + str(y_data[-1]))

    plt.savefig(os.path.join(output_dir, DIR_GRAPHICS, filename + '.png'))
    #plt.show()

def save_summary_model(output_dir, G, D, channels, noise_dim=100, img_size=28, is_conditional=False):
    f = io.StringIO()

    if channels == 1:
        G_in = (noise_dim,)
        D_in = (img_size**2,)
    elif channels == 3:
        G_in = (noise_dim,1,1,)
        D_in = (channels,img_size,img_size,)
    else:
        raise Exception("Number of channels unexpected: ", str(channels))

    with redirect_stdout(f):
        print("Saving model G/D summary")
        if is_conditional:
            summary(G.float(), [G_in, (10,)])
            summary(D.float(), [D_in, (10,)])
        else:
            summary(G.float(), [G_in])
            summary(D.float(), [D_in])
    out = f.getvalue()
    with open(os.path.join(output_dir, SUMMARY_FILE), 'w') as f:
        f.write(out)

def calculate_classifier_accuracy(generated_images, labels, classifier, dataset='MNIST'):
    if dataset == 'MNIST':
        predicted = classifier_accuracy.predict_from(generated_images, classifier)
        acc = classifier_accuracy.calculate_accuracy(labels, predicted)
        return acc
    else:
        raise Exception("Database different from MNIST not implemented yet for this metric in condGAN")

def model_converged(file_to_check, fnt):
    if 'npz' in file_to_check:
        array = np.load(file_to_check)
    elif 'csv' in file_to_check:
        with open(file_to_check, 'r') as f:
            array = list(csv.reader(f))
    else:
        raise Exception("Extension not implemented yet, file is:", file_to_check)
    return fnt(array)

def load_metric_evaluation(model_path, metric_file='pre_classifier_accuracies.csv', threshold_convergence=0, less_than=False):
    metrics = []
    for seed in os.listdir(model_path):
        if not os.path.isdir(os.path.join(model_path, seed)): continue
        if not os.path.exists(os.path.join(model_path, seed, DIR_STATS, metric_file)): continue
        filepath = os.path.join(model_path, seed, DIR_STATS, metric_file)
        with open(filepath) as f:
            reader = csv.reader(f)
            values = np.array([float(x) for x in next(reader)])
            if less_than:
                if(min(values) <= threshold_convergence):
                    metrics.append(values)
            else:
                if(max(values) >= threshold_convergence):
                    metrics.append(values)
    return np.array(metrics)

def load_stats(directory, metric_file='pre_classifier_accuracies.csv', threshold=0, less_than=False):
    return load_metric_evaluation(directory, metric_file=metric_file, threshold_convergence=threshold, less_than=less_than)
