import torch
import os

# ------------
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
N_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float
torch.set_default_dtype(torch.float32)
DIR_MODEL_CHECKPOINT = 'model_checkpoints'
# ------------

def configure_dtype(config):
    global DTYPE
    if(config["DOUBLE_TENSORS"]):
        DTYPE = torch.double
        torch.set_default_dtype(torch.double)

def set_seed_as(value):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)

def noise(size, dim, channels=1):
    if channels == 1:
        t = torch.randn(size, dim, dtype=DTYPE).to(DEVICE)
    elif channels == 3:
        t = torch.randn(size, dim, 1, 1, dtype=DTYPE).to(DEVICE)
    else:
        raise Exception("Still not implemented the following case: channels=" + str(channels))
    return t

def fake_labels_balanced_ordered(N, n_classes, double_tensors):
    if N % n_classes != 0: raise Exception("n_classes should be factor of N (N %% n_classes is not zero)")
    samples = []
    for i in range(n_classes):
        for j in range(int(N/n_classes)):
            c = [0] * n_classes
            c[i] = 1
            samples.append(c)
    samples = torch.tensor(samples).to(DEVICE)
    if(double_tensors): samples = samples.double()
    return samples


def fake_labels(N, double_tensors):
    dist = torch.distributions.one_hot_categorical.OneHotCategorical(torch.tensor([ 0.1 ]*10))
    samples = dist.sample(torch.Size([N])).to(DEVICE)
    if(double_tensors): samples = samples.double()
    return samples

def ones_target(size, smooth=True):
    t = torch.ones(size, 1, dtype=DTYPE).to(DEVICE)
    if(smooth): t[:] = 0.9
    return t

def zeros_target(size):
    t = torch.zeros(size, 1, dtype=DTYPE).to(DEVICE)
    return t

def images_to_vectors(images, dim=784):
    return images.view(images.size(0), dim)

def vectors_to_images(vectors, dim=28, channels=1):
    return vectors.view(vectors.size(0), channels, dim, dim)

def one_hot_vector(labels, n_classes=10):
    N = labels.size(0)

    labels = labels.view(-1, 1)
    onehot = torch.FloatTensor(N, n_classes).to(DEVICE)

    onehot.zero_()
    onehot.scatter_(1, labels, 1)

    return onehot

def model_checkpoint(epoch, config, G, D, d_optim, g_optim, output_dir):
    print('saving')
    torch.save({
        'epoch': epoch,
        'config': config,
        'gen_state_dict': G.state_dict(),
        'dis_state_dict': D.state_dict(),
        'gen_optimizer': g_optim.state_dict(),
        'dis_optimizer': d_optim.state_dict()
    }, os.path.join(output_dir, DIR_MODEL_CHECKPOINT, str(epoch)))
    print('done')
