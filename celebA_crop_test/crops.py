from torchvision import transforms, datasets
from torchvision.utils import save_image
import torch
import helper

def get_celebA_data(train, img_size=64, crop=178):
    compose = transforms.Compose(
            [
                transforms.CenterCrop(crop),
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
    out_dir = '../data'
    return datasets.CelebA(root=out_dir,split='train' if train else 'test',transform=compose,download=True)

dataset_64 = get_celebA_data(True, 64)
dataset_128 = get_celebA_data(True, 128)
dataloader_64 = torch.utils.data.DataLoader(dataset_64, batch_size=32, shuffle=True)
dataloader_128 = torch.utils.data.DataLoader(dataset_128, batch_size=32, shuffle=True)
images_64, _ = next(iter(dataloader_64))
images_128, _ = next(iter(dataloader_128))

save_image(images_64[:25], 'celebA_crop_test/celeba_64.png', nrow=5, normalize=True)
save_image(images_128[:25], 'celebA_crop_test/celeba_128.png', nrow=5, normalize=True)
