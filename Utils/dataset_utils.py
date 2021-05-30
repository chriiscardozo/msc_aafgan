import torch
from torchvision import transforms, datasets
from Utils import commons_utils
import os, json, csv

MANUAL_CLASSIFICATION = 'manual_classification.json'
MODEL_MAPPING = None


def get_model_mapping_value(key):
    global MODEL_MAPPING
    count = 0
    if MODEL_MAPPING is None:
        MODEL_MAPPING = {}
        for d_model in ['default', 'BHSA', 'BHAA']:
            for g_model in ['default', 'BHSA', 'BHAA']:
                for d_hidden_model in ['default', 'SHReLU', 'MiDA']:
                    for g_hidden_model in ['default', 'SHReLU', 'MiDA']:
                        MODEL_MAPPING[str(count)] = { 'd_model': d_model, 'g_model': g_model,
                                                 'd_hidden_model': d_hidden_model, 'g_hidden_model': g_hidden_model }
                        count += 1
    return MODEL_MAPPING[key]


def get_mnist_data(train, img_size=28):
    compose = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, ), std=(0.5, ))
            ])
    out_dir = './data'
    return datasets.MNIST(root=out_dir,train=train,transform=compose,download=True)


def get_cifar10_data(train, img_size=64):
    compose = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
    out_dir = './data'
    return datasets.CIFAR10(root=out_dir,train=train,transform=compose,download=True)


def get_celebA_data(train, img_size=64):
    compose = transforms.Compose(
            [
                transforms.CenterCrop(178),
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
    out_dir = './data'
    return datasets.CelebA(root=out_dir,split='train' if train else 'test',transform=compose,download=False)


def get_train_test_data(config, dataset='MNIST'):
    if dataset == 'MNIST':
        train_data = get_mnist_data(True, config['IMG_SIZE'])
        test_data = get_mnist_data(False, config['IMG_SIZE'])
    elif dataset == 'CIFAR10':
        train_data = get_cifar10_data(True, config['IMG_SIZE'])
        test_data = get_cifar10_data(False, config['IMG_SIZE'])
    elif dataset == 'CelebA':
        train_data = get_celebA_data(True, config['IMG_SIZE'])
        test_data = get_celebA_data(False, config['IMG_SIZE'])
    else:
        raise Exception("dataset load not implemented for dataset '" + dataset + "'")
    
    # PIN_MEMORY speeds up CPU to GPU dataset transfer
    pin_memory = True if 'PIN_MEMORY' in config and int(config['PIN_MEMORY']) == 1 else False
    # NUM_WORKERS rely on CPU cores amount
    X_train = torch.utils.data.DataLoader(train_data, batch_size=config['BATCH_SIZE'], shuffle=True,
                                          pin_memory=pin_memory, num_workers=int(commons_utils.cpu_count()/2))
    X_test = torch.utils.data.DataLoader(test_data, batch_size=len(test_data), shuffle=True, pin_memory=pin_memory, num_workers=int(commons_utils.cpu_count()/2))
    if dataset == 'MNIST': X_test = next(iter(X_test))[0].view(len(test_data), -1)
    else: X_test = next(iter(X_test))[0]

    return X_train, X_test


def get_manually_classified_data(path):
    X = []
    y = []

    file_path = os.path.join(path, MANUAL_CLASSIFICATION)

    if os.path.exists(file_path):
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
            with open(os.path.join(path, commons_utils.DIR_SAMPLES_CSV, 'samples_199.csv')) as f:
                samples = list(csv.reader(f))

                for key in data.keys():
                    X.append([float(v) for v in samples[int(key)]])
                    y.append(int(data[key]))

            if y.count(0) == len(data.keys()):
                X = []
                y = []

    return X, y
