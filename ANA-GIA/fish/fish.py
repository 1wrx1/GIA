try:
    import breaching
except ModuleNotFoundError:
    # You only really need this safety net if you want to run these notebooks directly in the examples directory
    # Don't worry about this if you installed the package or moved the notebook to the main directory.
    import os; os.chdir("..")
    import breaching
    
import torch
import matplotlib.pyplot as plt

import logging, sys

import lpips
import metrics

import os
from PIL import Image
import torch
from torchvision import transforms

import numpy as np

import argparse

from util import grid_plot


parser = argparse.ArgumentParser(description='Fishing.')
parser.add_argument('--bs', type=int, default=64,
                    help='batch size')
parser.add_argument('--resolution', type=int, default=128,
                    help='resolution')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--pretrain', type=bool, default=True,
                    help='pretrain')
parser.add_argument('--arch', type=str, default='resnet18',
                    help='target model')
parser.add_argument('--root', type=str,
                    help='ImageNet root')
args = parser.parse_args()

bs = args.bs
res = args.resolution

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)], format='%(message)s')
logger = logging.getLogger()

cfg = breaching.get_config(overrides=["case/server=malicious-fishing", "attack=clsattack", "case/user=multiuser_aggregate"])
          
device = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() else torch.device('cpu')
torch.backends.cudnn.benchmark = cfg.case.impl.benchmark
setup = dict(device=device, dtype=getattr(torch, cfg.case.impl.dtype))

folder_path = 'custom_data/imgs'

image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]

image_files = sorted(image_files, key=lambda name: int(name.split('-')[0]))

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

s = torch.tensor(std).view(3, 1, 1)

dm = torch.as_tensor(mean)[:, None, None]
ds = torch.as_tensor(std)[:, None, None]

# 定义图片的预处理转换（例如，将图像转换为Tensor并归一化）
transform = transforms.Compose([
    transforms.Resize(res),
    transforms.ToTensor(),  # 将图像转换为张量，并自动将值从 [0, 255] 归一化到 [0.0, 1.0]
    transforms.Normalize(mean=mean, std=std)
])

# 创建一个列表来存储所有图片的tensor
image_tensors = []
labels_list = []

# 逐个读取图片并转换为tensor
for image_file in image_files:

    label = int(image_file.split('-')[0])

    labels_list.append(label)
    
    image_path = os.path.join(folder_path, image_file)
    image = Image.open(image_path)
    image_tensor = transform(image)
    image_tensors.append(image_tensor)

image_batch_tensor = torch.stack(image_tensors).to(device)
labels = torch.tensor(labels_list).to(device)

recs = []
psnrs, ssims, lpipss = [], [], []

for j in range(64 // bs):

    custom_data = {'inputs':image_batch_tensor[j*bs:(j+1)*bs], 'labels':labels[j*bs:(j+1)*bs]}

    for i in range(bs):
        label = labels_list[i]
        
        cfg.case.user.user_range = [0, 1]
    
        cfg.case.data.partition = "random"
        cfg.case.user.num_data_points = bs
        cfg.case.data.default_clients = 32
        
        cfg.case.user.provide_labels = True
        cfg.case.server.target_cls_idx = i
    
        cfg.case.data.path = args.root
        cfg.case.data.name = 'ImageNet'
        cfg.case.data.classes = 1000
        cfg.case.data.shape = [3, res, res]
        cfg.case.data.augmentations_val = {'Resize': res}
        cfg.case.model = args.arch
        cfg.case.server.pretrained = args.pretrain
    
        user, server, model, loss_fn = breaching.cases.construct_case(cfg.case, setup)
        attacker = breaching.attacks.prepare_attack(server.model, server.loss, cfg.attack, setup)
    
        attacker.cfg.optim.max_iterations = 24000
        
        breaching.utils.overview(server, user, attacker)
    
        user.users[0].custom_data = custom_data
    
        [shared_data], [server_payload], true_user_data = server.run_protocol(user)
    
        reconstructed_user_data, stats = attacker.reconstruct([server_payload], [shared_data], 
                                                          server.secrets, dryrun=cfg.dryrun)
        fished_data = dict(data=reconstructed_user_data["data"][server.secrets["ClassAttack"]["target_indx"]], 
                       labels=None)
    
        ground_truth = true_user_data['data'].clone().cpu()
        x = ground_truth[i].unsqueeze(0)
    
        rec = fished_data['data'].clone().cpu()
    
        lpips_scorer = lpips.LPIPS(net="alex")
    
        ssim_score = metrics.cw_ssim(rec, x, scales=5)
        psnr_score = metrics.psnr(rec, x, factor=1 / s)
        lpips_score = lpips_scorer(rec, x, normalize=True)
    
        psnrs.append(torch.tensor(psnr_score, dtype=torch.float))
        ssims.append(torch.tensor(ssim_score, dtype=torch.float))
        lpipss.append(torch.mean(lpips_score))
    
        rec = rec.mul_(torch.as_tensor(ds)).add_(torch.as_tensor(dm))
    
        tensor = rec.clone().detach().cpu()
        
        recs.append(tensor)

output_all = torch.cat(recs, dim=0)

print(output_all.shape)

grid_plot(output_all, bs, args)

psnrs = torch.stack(psnrs)
ssims = torch.stack(ssims)
lpipss = torch.stack(lpipss)

print(f'PSNR: {torch.mean(psnrs)} +- {torch.std(psnrs)}')
print(f'SSIM: {torch.mean(ssims)} +- {torch.std(ssims)}')
print(f'lpips: {torch.mean(lpipss)} +- {torch.std(lpipss)}')

    

    