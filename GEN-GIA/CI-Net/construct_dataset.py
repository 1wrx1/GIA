from PIL import Image
import pandas as pd
import os
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.data_frame = pd.read_csv(os.path.join(self.root_dir, csv_file))

    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir, 'selected_images', self.data_frame.iloc[index, 0])
        img = Image.open(img_name).convert('RGB')
        img = self.transform(img)

        return img, self.data_frame.iloc[index, 1]

    def __len__(self):
        return len(self.data_frame)


def construct_dataset(dataset_name, dataset_root, download=True):
    if dataset_name == 'cifar10':
        mean = [0.4914672374725342, 0.4822617471218109, 0.4467701315879822]
        std = [0.24703224003314972, 0.24348513782024384, 0.26158785820007324]

        classes = 10

        size = 32

        dataset = datasets.CIFAR10(
                root=dataset_root, download=download, train=False,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]))

    elif dataset_name == 'cifar100':
        mean = [0.5071598291397095, 0.4866936206817627, 0.44120192527770996]
        std = [0.2673342823982239, 0.2564384639263153, 0.2761504650115967]

        classes = 100

        size = 32
        
        dataset = datasets.CIFAR100(
                root=dataset_root, download=download, train=False,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]))

    elif dataset_name == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        classes = 1000

        size = 128
        
        dataset = datasets.ImageNet(
                root=dataset_root, split='val',
                transform=transforms.Compose([
                    transforms.Resize((128, 128)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]))

    elif dataset_name == 'imagenet64':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        classes = 1000

        size = 64
        
        dataset = datasets.ImageNet(
                root=dataset_root, split='val',
                transform=transforms.Compose([
                    transforms.Resize((64, 64)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]))

    elif dataset_name == 'imagenet256':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        classes = 1000

        size = 256
        
        dataset = datasets.ImageNet(
                root=dataset_root, split='val',
                transform=transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]))

    elif dataset_name == 'celeba64':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

        classes = 1000

        size = 64
        
        transform = transforms.Compose([
                                        transforms.Resize((64, 64)),
                                        transforms.CenterCrop(64),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std),
                                        ])
        
        dataset = ImageDataset(dataset_root, 'selected_identities.csv', transform)
        
    else:
        raise NotImplementedError("Not implemented dataset.")
        

    return dataset, mean, std, classes, size