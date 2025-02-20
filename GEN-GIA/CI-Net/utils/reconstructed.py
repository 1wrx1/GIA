from __future__ import print_function

import argparse
import os
import re
import random 
import torch
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.checkpoint as cp
import torchvision.utils as vutils
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import copy
import warnings
import subprocess
import random


from functools import partial
from copy import deepcopy
from typing import Type, Any, Callable, Union, List, Optional, Tuple
from collections import OrderedDict, defaultdict
from tqdm.notebook import tqdm, trange

DEBUG = False


def convert_relu_to_sigmoid(model):
    for child_name, child in model.named_children():
        if (isinstance(child, nn.ReLU)) or (isinstance(child, nn.GELU)):
            setattr(model, child_name, nn.Sigmoid())
        else:
            convert_relu_to_sigmoid(child)
            
def convert_relu_to_tanh(model):
    for child_name, child in model.named_children():
        if (isinstance(child, nn.ReLU)) or (isinstance(child, nn.GELU)):
            setattr(model, child_name, nn.Tanh())
        else:
            convert_relu_to_tanh(child)
            
def convert_relu_to_leaky(model):
    for child_name, child in model.named_children():
        if (isinstance(child, nn.ReLU)) or (isinstance(child, nn.GELU)):
            setattr(model, child_name, nn.LeakyReLU(0.1))
        else:
            convert_relu_to_leaky(child)
            
def convert_relu_to_rrelu(model):
    for child_name, child in model.named_children():
        if (isinstance(child, nn.ReLU)) or (isinstance(child, nn.GELU)):
            setattr(model, child_name, nn.RReLU())
        else:
            convert_relu_to_rrelu(child)
            
def convert_relu_to_gelu(model):
    for child_name, child in model.named_children():
        if (isinstance(child, nn.ReLU)) or (isinstance(child, nn.GELU)):
            setattr(model, child_name, nn.GELU())
        else:
            convert_relu_to_gelu(child)

#apply for initialize the model
def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)
def query_mem():
    total_memory,used_memory, free_memory = map(int,os.popen('free -t -m' ).readlines()[-1].split()[1:])
    print("Memory Usage:", used_memory,"Total memory",total_memory)
    return used_memory


def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map  


def patch_consistency(X, P):
    """
    计算给定批量图像的patch consistency loss。

    参数:
        X (torch.Tensor): 输入图像张量，形状为 (B, 3, H, W)。
        P (int): Patch 的大小。

    返回:
        loss (torch.Tensor): 计算出的patch consistency loss，形状为 (B,)。
    """
    B, C, H, W = X.shape
    loss = torch.tensor(0.0, device=X.device)

    # 计算垂直方向的patch差异
    for k in range(1, H // P):
        patch_current = X[:, :, P * k:P * k + 1, :]
        patch_previous = X[:, :, P * k - 1:P * k, :]
        loss += torch.norm(patch_current - patch_previous)

    # 计算水平方向的patch差异
    for k in range(1, W // P):
        patch_current = X[:, :, :, P * k:P * k + 1]
        patch_previous = X[:, :, :, P * k - 1:P * k]
        loss += torch.norm(patch_current - patch_previous)

    return loss

    
# calculating the gradient loss based on the cosine similarity + total variation + (patch_consistency)
def loss_inverting_gt(original_dy_dx,fake_dy_dx,fake_img,tv_value, pc_value=0, idx=None):
    errG = 0
    pnorm = [0,0]

    total_length = len(fake_dy_dx)

    if idx is not None:
        grad_idx = idx
    else:
        grad_idx = total_length

    for i in range(grad_idx):
        gx = fake_dy_dx[i]
        gy = original_dy_dx[i]
    #for gx,gy in zip(fake_dy_dx, original_dy_dx):
        errG -= (gx * gy).sum()
        pnorm[0] += gx.pow(2).sum()
        pnorm[1] += gy.pow(2).sum() 
        
    errG = (1 + errG / pnorm[0].sqrt() / pnorm[1].sqrt())
    
    errG+= tv_value * total_variation(fake_img)

    if pc_value > 0:
        errG += pc_value * patch_consistency(fake_img, 16)
        
    return errG


# calculating the gradient loss based on the L2 norm only
def loss_l2(original_dy_dx,fake_dy_dx):
    errG = 0
    for gx,gy in zip(fake_dy_dx, original_dy_dx):
        errG += ((gx - gy) ** 2).sum()
    return errG


def total_variation(x):
    """Total_variation : compute the tv value for inverting gradient and Gan attack"""
    """Anisotropic TV."""
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy
