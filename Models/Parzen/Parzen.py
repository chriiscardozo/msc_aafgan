import csv
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from Utils import cuda_utils
import torch

global cpu_kde, cpu_kde_conditional
cpu_kde = None
cpu_kde_conditional = {}

def build_cpu_kde(X):
    # Cross-validation to find best bandwidth
    params = {'bandwidth': np.linspace(0.01, 10, 20)} 
    grid = GridSearchCV(KernelDensity(kernel='gaussian'), params, n_jobs=8)
    grid.fit(X)
    print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
    return grid.best_estimator_

def log_prob(X_test, samples, gpu=False, class_code=None):
    if(gpu):
        raise Exception("TODO: Parzen KDE na GPU não implementada ainda")
        return gpu_log_prob(X_test,samples)
    else:
        return cpu_log_prob(X_test,samples, class_code)

def cpu_log_prob(X_test, samples, class_code=None):
    if(cuda_utils.DEVICE != 'cpu'):
        cpu_X = X_test.clone()
        cpu_X = cpu_X.cpu().numpy()
        cpu_samples = samples.clone()
        cpu_samples = cpu_samples.cpu().numpy()
    else:
        cpu_X = X_test.numpy()
        cpu_samples = samples.numpy()
    
    global cpu_kde
    global cpu_kde_conditional

    if(class_code is None and cpu_kde is None):
        print("build the kde estimator")
        cpu_kde = build_cpu_kde(cpu_X)
    elif(class_code is not None and class_code not in cpu_kde_conditional):
        print("build the kde estimator for the class", class_code)
        cpu_kde_conditional[class_code] = build_cpu_kde(cpu_X)
    # else:
    #     print("cpu_kde/cpu_kde_conditional is not None... moving forward")
    
    current_kde = cpu_kde if class_code is None else cpu_kde_conditional[class_code]

    scores = current_kde.score_samples(cpu_samples)

    return [np.mean(scores), np.std(scores)] # return mean log prob and std log prob

def gpu_log_prob(X_test,samples):
    lls_avg_sigma = []
    lls_std_sigma = []
    params = {'bandwidth': np.linspace(0.01, 10, 20)}
    for sigma in params['bandwidth']:
        ll_avg, ll_std = _gpu_ll(samples, X_test, sigma)
        lls_avg_sigma.append(ll_avg)
        lls_std_sigma.append(ll_std)
    index = np.array(lls_avg_sigma).argmax()
    print("best bandwidth:", params['bandwidth'][index])
    return [lls_avg_sigma[index], lls_std_sigma[index]]

def _gpu_ll(x, mu, sigma):
    p = torch.sum(_gpu_gaussian_kernel(x-mu)/sigma)

def _gpu_gaussian_kernel(x, sigma):
    num = x*x
    den = 2*(sigma*sigma)
    return torch.exp(-(num/den))
