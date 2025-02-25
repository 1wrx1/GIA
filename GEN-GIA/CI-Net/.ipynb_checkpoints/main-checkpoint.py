from __future__ import print_function

import argparse
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torchvision
import torch.utils.data
import matplotlib.pyplot as plt
import warnings 
from vit_pytorch import SimpleViT
from torchvision import datasets, transforms
from model.model_architectures import construct_model, CustomVitTimm, train_model
from utils.reconstructed import convert_relu_to_sigmoid, convert_relu_to_tanh, convert_relu_to_leaky, \
    convert_relu_to_rrelu, convert_relu_to_gelu
from utils.config import config
from utils.nb_utils import compute_batch_order, grid_plot
from model.generator import trainer
import torchvision.models as models
import metrics
import lpips
import os
from argparse import ArgumentParser
from scipy.optimize import linear_sum_assignment
import timm
import numpy as np
from vit_LoRA import SimpleViT_LoRA
from construct_dataset import construct_dataset


warnings.filterwarnings("ignore")
 
parser = ArgumentParser("CI-Net")
parser.add_argument("--dataset", default='cifar100', type=str, help='target dataset')
parser.add_argument("--root", type=str, help='dataset root')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--bs', type=int, default=64, help='batch size')
parser.add_argument('--grad_idx', type=int, default=1, help='usage of gradients')
parser.add_argument('--patch', type=float, default=1e-3, help='patch consistency')
parser.add_argument("--act", default='sigmoid', type=str, help='activation layers')
parser.add_argument("--arch", default='resnet18', type=str, help='architecture of the target model')
parser.add_argument("--defense", default='no_defense', type=str, help='defense')
parser.add_argument('--noise', type=float, default=0., help='noise std')
parser.add_argument('--p', type=float, default=0., help='p for prun or soteria')

parser.add_argument('--pretrain', type=bool, default=False, help='target model pretrained or not')


args = parser.parse_args()

act = args.act
size_batch = args.bs

config['device'] = f'cuda:{args.gpu}'
config['dst'] = args.dataset
config['defense'] = args.defense

if config['defense'] == 'noise':
    print(args.noise)
    config['noise_std'] = args.noise    
elif config['defense'] == 'prun':
    config['prun_p'] = args.p    
elif config['defense'] == 'soteria':
    config['soteria_p'] = args.p    
else:
    pass
    
if 'LoRA' in args.arch:
    config['LoRA'] = True    
else:
    config['LoRA'] = False

dataset, mean, std, classes, size = construct_dataset(config['dst'], args.root)

config['classes'] = classes

idx = 48

label_all, used_idx, imgs = [], [], []

psnrs, ssims, lpipss = [], [], []

while len(label_all) < 64:
    img, label = dataset[idx]

    if args.dataset != 'cifar10':           
        if label not in label_all:    
            label_all.append(label)
            used_idx.append(idx)
            imgs.append(img)

    else:
        label_all.append(label)
        used_idx.append(idx)
        imgs.append(img)

    idx += 1

x = torch.stack(imgs)

exps = 64 // size_batch

rec_images = []

for exp_i in range(exps):
    if 'resnet' in args.arch.lower():
        model = train_model(arch=args.arch, num_classes=config['classes'], pretrained=args.pretrain)
    elif args.arch == 'vit-base':
        model = CustomVitTimm(scale='base', img_size=size, num_classes=config['classes'], pretrained=args.pretrain)
        config['patch'] = args.patch
    elif args.arch == 'vit-tiny':
        model = CustomVitTimm(scale='tiny', img_size=size, num_classes=config['classes'], pretrained=args.pretrain)
        config['patch'] = args.patch
    elif args.arch == 'vit-large-LoRA':
        model =  SimpleViT_LoRA(
            image_size=size,
            patch_size=16,
            num_classes=config['classes'],
            dim=1024,
            depth=24,
            heads=16,
            mlp_dim=4096
        )
        model.freeze()
    elif args.arch == 'vit-base-LoRA':
        model =  SimpleViT_LoRA(
            image_size=size,
            patch_size=16,
            num_classes=config['classes'],
            dim=768,
            depth=12,
            heads=12,
            mlp_dim=3072
        )
        model.freeze()
    elif args.arch == 'vit-tiny-LoRA':
        model =  SimpleViT_LoRA(
            image_size=size,
            patch_size=16,
            num_classes=config['classes'],
            dim=192,
            depth=12,
            heads=3,
            mlp_dim=768
        )
        model.freeze()
    elif args.arch == 'vit-huge-LoRA':
        model =  SimpleViT_LoRA(
            image_size=size,
            patch_size=16,
            num_classes=config['classes'],
            dim=1280,
            depth=32,
            heads=16,
            mlp_dim=5120
        )
        model.freeze()    

    model = model.to(config['device'])

    if act == 'relu':
        pass
    elif act == 'sigmoid':
        convert_relu_to_sigmoid(model)
    elif act == 'tanh':
        convert_relu_to_tanh(model)
    elif act == 'leaky':
        convert_relu_to_leaky(model)
    elif act == 'rrelu':
        convert_relu_to_rrelu(model)
    elif act == 'gelu':
        convert_relu_to_gelu(model)

    print(model)

    config["nz"] = 128
    config['total_img'] = size_batch
    config['b_size'] = size_batch
    if size_batch in [32, 64]:
        config['num_epochs'] = 5000
    else:
        config['num_epochs'] = 2000

    config['rep_freq'] = config['num_epochs'] / 100

    attack = {'method': 'CI-Net', 'lr': 0.001}
    config['lr'] = attack['lr']
    trainer_ = trainer(config, attack, model, dataset, used_idx[size_batch * exp_i:size_batch * (exp_i + 1)])

    reconstructed_image, avg_score = trainer_.attack_training()

    lpips_scorer = lpips.LPIPS(net="alex")

    x_rec = reconstructed_image["0"]["image"]['CI-Net']['ssim']["timeline"][0][-1]

    order = compute_batch_order(lpips_scorer, x_rec, x[size_batch * exp_i:size_batch * (exp_i + 1)])
    x_rec = x_rec[order]

    std_ = torch.tensor(std).view(3, 1, 1)

    test_psnr = metrics.psnr(x_rec, x[size_batch * exp_i:size_batch * (exp_i + 1)], factor=1 / std_)
    test_ssim = metrics.cw_ssim(x_rec, x[size_batch * exp_i:size_batch * (exp_i + 1)], scales=5)
    lpips_score = lpips_scorer(x_rec, x[size_batch * exp_i:size_batch * (exp_i + 1)], normalize=True)

    psnrs.append(torch.tensor(test_psnr, dtype=torch.float))
    ssims.append(torch.tensor(test_ssim, dtype=torch.float))
    lpipss.append(torch.mean(lpips_score))

    dm = torch.as_tensor(mean)[:, None, None]
    ds = torch.as_tensor(std)[:, None, None]
    
    x_rec = x_rec.mul_(torch.as_tensor(ds)).add_(torch.as_tensor(dm))

    tensor = x_rec.clone().detach().cpu()
    
    rec_images.append(tensor)
    
output_all = torch.cat(rec_images, dim=0)

grid_plot(output_all, args, ds, dm)

psnrs = torch.stack(psnrs)
ssims = torch.stack(ssims)
lpipss = torch.stack(lpipss)

print(f'PSNR: {torch.mean(psnrs)} +- {torch.std(psnrs)}')
print(f'SSIM: {torch.mean(ssims)} +- {torch.std(ssims)}')
print(f'lpips: {torch.mean(lpipss)} +- {torch.std(lpipss)}')