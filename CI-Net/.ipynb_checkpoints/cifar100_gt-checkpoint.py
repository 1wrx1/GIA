from __future__ import print_function

import argparse
import time
import torch
import torch.nn as nn
import torch.nn.parallel

import torch.utils.data
import matplotlib.pyplot as plt
import warnings 

from vit_pytorch import SimpleViT

from torchvision import datasets, transforms
from model.model_architectures import construct_model
from utils.reconstructed import convert_relu_to_sigmoid, convert_relu_to_tanh, convert_relu_to_leaky, \
    convert_relu_to_rrelu, convert_relu_to_gelu
from utils.config import config
from model.generator import trainer
import torchvision.models as models
import metrics
import lpips

import os
from argparse import ArgumentParser

from scipy.optimize import linear_sum_assignment
import timm

import numpy as np

warnings.filterwarnings("ignore")


class CustomVitTimm(nn.Module):
    def __init__(self, scale='base', img_size=64, num_classes=1000, pretrained=False):
        super().__init__()
        # Load the pre-trained ViT model
        self.model = timm.create_model(f'vit_{scale}_patch16_224.augreg_in21k', pretrained=pretrained)

        embed_dim_dict = {'tiny': 192, 'small': 384, 'base': 768, 'large': 1024}

        # Modify the patch embedding layer
        # Original size: (224, 224) with 16x16 patches => (14, 14) grid of patches
        # New size: (64, 64) with 16x16 patches => (4, 4) grid of patches
        self.model.patch_embed = timm.layers.PatchEmbed(
            img_size=img_size,
            patch_size=16,
            in_chans=3,
            embed_dim=embed_dim_dict[scale]
        )

        # Adjusting the position embeddings since the number of patches has changed
        # Original: 14*14 + 1 = 197, New: 4*4 + 1 = 17
        num_patches = (img_size // 16) * (img_size // 16)
        self.model.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim_dict[scale]))
        nn.init.normal_(self.model.pos_embed, std=0.02)  # Initializing weights as per original ViT

        # Adjusting the classifier head
        self.model.head = nn.Linear(embed_dim_dict[scale], num_classes)

    def forward(self, x):
        return self.model(x)
        
        
def normalize_tensor(in_feat, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat ** 2, dim=1, keepdim=True))
    return in_feat / (norm_factor + eps)

def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2, 3], keepdim=keepdim)

def compute_batch_order(lpips_scorer, output, ground_truth):
    """Re-order a batch of images according to LPIPS statistics of source batch, trying to match similar images.
    This implementation basically follows the LPIPS.forward method, but for an entire batch."""

    B = output.shape[0]
    L = lpips_scorer.L
    assert ground_truth.shape[0] == B

    with torch.inference_mode():
        # Compute all features [assume sufficient memory is a given]
        features_rec = []
        for input in output:
            input_scaled = lpips_scorer.scaling_layer(input)
            output = lpips_scorer.net.forward(input_scaled)
            layer_features = {}
            for kk in range(L):
                layer_features[kk] = normalize_tensor(output[kk])
            features_rec.append(layer_features)

        features_gt = []
        for input in ground_truth:
            input_scaled = lpips_scorer.scaling_layer(input)
            output = lpips_scorer.net.forward(input_scaled)
            layer_features = {}
            for kk in range(L):
                layer_features[kk] = normalize_tensor(output[kk])
            features_gt.append(layer_features)

        # Compute overall similarities:
        similarity_matrix = torch.zeros(B, B)
        for idx, x in enumerate(features_gt):
            for idy, y in enumerate(features_rec):
                for kk in range(L):
                    diff = (x[kk] - y[kk]) ** 2
                    similarity_matrix[idx, idy] += spatial_average(lpips_scorer.lins[kk](diff)).squeeze()
    try:
        _, rec_assignment = linear_sum_assignment(similarity_matrix.cpu().numpy(), maximize=False)
    except ValueError:
        print(f"ValueError from similarity matrix {similarity_matrix.cpu().numpy()}")
        print("Returning trivial order...")
        rec_assignment = list(range(B))
    return torch.as_tensor(rec_assignment, dtype=torch.long)


def grid_plot(tensor, args, ds, dm):
    tensor = tensor.clone().detach()
    tensor.mul_(ds).add_(dm).clamp_(0, 1)
    num_images = 64
    if num_images == 1:
        fig, axes = plt.subplots(1, 1, figsize=(1, 1))
    elif num_images in [4, 8]:
        fig, axes = plt.subplots(1, num_images, figsize=(num_images, num_images))
    else:
        fig, axes = plt.subplots(num_images // 8, 8, figsize=(12, num_images // 16 * 3))
    axes = np.reshape(axes, -1)
    for im, ax in zip(tensor, axes):
        ax.imshow(im.permute(1, 2, 0).cpu())
        ax.axis('off')
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    os.makedirs('saved_img') if not os.path.exists('saved_img') else None
    file_name = 'saved_img/' + 'cifar10_gt' + '.pdf'
    plt.savefig(file_name)
    

parser = ArgumentParser("CI-Net")
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--bs', type=int, default=64, help='batch size')
parser.add_argument("--act", default='relu', type=str, help='activation layers')
parser.add_argument("--arch", default='resnet18', type=str, help='architecture of the target model')
#parser.add_argument("--normalize", default=False, type=bool, help='normalize or not')


args = parser.parse_args()


#act = 'relu'
act = args.act

size_batch = args.bs

config['device'] = f'cuda:{args.gpu}'
config['classes'] = 10
config['dst'] = "cifar10"

#size_batch = 64

#cifar100_mean = [0.5071598291397095, 0.4866936206817627, 0.44120192527770996]
#cifar100_std = [0.2673342823982239, 0.2564384639263153, 0.2761504650115967]
cifar10_mean = [0.4914672374725342, 0.4822617471218109, 0.4467701315879822]
cifar10_std = [0.24703224003314972, 0.24348513782024384, 0.26158785820007324]


dataset = datasets.CIFAR10(
        root='/grp01/saas_medml/runxi/cifar10', download=True, train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ]))

idx = 48

label_all, used_idx, imgs = [], [], []

psnrs, ssims, lpipss = [], [], []

while len(label_all) < 64:
    img, label = dataset[idx]
    #print(label)

    label_all.append(label)
    used_idx.append(idx)
    imgs.append(img)
        
    idx += 1
        
#print(used_idx)
x = torch.stack(imgs)

dm = torch.as_tensor(cifar10_mean)[:, None, None]
ds = torch.as_tensor(cifar10_std)[:, None, None]


grid_plot(x, args, ds, dm)