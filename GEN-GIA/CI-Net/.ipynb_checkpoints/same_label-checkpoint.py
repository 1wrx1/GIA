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

from vit_LoRA import SimpleViT_LoRA

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
    #tensor.mul_(ds).add_(dm).clamp_(0, 1)
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
    
    if 'vit' in args.arch:
        file_name = 'saved_img/' + args.arch + '_' + 'cifar10_' + 'bs_' + str(args.bs) + '_act_' + args.act + '_patch_' + str(args.patch) + '.pdf'
    else:
        if args.defense == 'no_defense':
            file_name = 'saved_img/' + args.arch + '_' + 'cifar10_' + 'bs_' + str(args.bs) + '_act_' + args.act + '_no_defense_' + '.pdf'
        elif args.defense == 'noise':
            file_name = 'saved_img/' + args.arch + '_' + 'cifar10_' + 'bs_' + str(args.bs) + '_act_' + args.act + '_noise_' + str(args.noise) + '.pdf'
        elif args.defense == 'prun':
            file_name = 'saved_img/' + args.arch + '_' + 'cifar10_' + 'bs_' + str(args.bs) + '_act_' + args.act + '_prun_' + str(args.p) + '.pdf'
        elif args.defense == 'soteria':
            file_name = 'saved_img/' + args.arch + '_' + 'cifar10_' + 'bs_' + str(args.bs) + '_act_' + args.act + '_soteria_' + str(args.p) + '.pdf'
        elif args.defense == 'no_fc':
            file_name = 'saved_img/' + args.arch + '_' + 'cifar10_' + 'bs_' + str(args.bs) + '_act_' + args.act + '_no_fc_' + '.pdf'
        elif args.defense == 'adapt-noise':
            file_name = 'saved_img/' + args.arch + '_' + 'cifar10_' + 'bs_' + str(args.bs) + '_act_' + args.act + '_adapt_noise_' + 'p_' + str(args.p) + '_noise_' + str(args.noise) + '.pdf'
    plt.savefig(file_name)
    

class train_model(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super(train_model, self).__init__()
        # 加载预训练的ResNet18模型
        self.resnet18 = torchvision.models.resnet18(pretrained=pretrained)
        
        self.resnet18.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet18.maxpool = nn.Sequential()

        self.resnet18.fc = nn.Sequential()
        
        # 提取ResNet18模型的卷积层（去掉最后的全连接层）
        #self.features = nn.Sequential(*list(self.resnet18.children())[:-2])  # 去掉最后的全局平均池化层和FC层

        #print(self.features)

        #print(self.resnet18)
        #self.avgpool = self.resnet18.avgpool  # 保留全局平均池化层
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)  # 保留全连接层，用于最终分类
    
    def get_feature(self, x):
        #out = self.features(x)

        #out = self.avgpool(out)
        out = self.resnet18(x)
        
        feature_map = torch.flatten(out, 1)
        
        return feature_map
        
    def forward(self, x):
        # 提取中间的特征图
        #out = self.features(x)

        #out = self.avgpool(out)

        #print(out.shape)
        out = self.resnet18(x)

        out = torch.flatten(out, 1)

        #print(feature_map.shape)
        
        #feature_map = out.view(out.size(0), -1)
        
        # 全局平均池化
        #pooled_features = self.avgpool(feature_map)
        #pooled_features = torch.flatten(feature_map, 1)
        
        # 最终分类结果
        output = self.fc(out)
        
        # 返回特征图和分类结果
        return output
    

parser = ArgumentParser("CI-Net")
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--bs', type=int, default=64, help='batch size')
parser.add_argument('--grad_idx', type=int, default=1, help='usage of gradients')
parser.add_argument('--patch', type=float, default=1e-3, help='patch consistency')
parser.add_argument("--act", default='relu', type=str, help='activation layers')
parser.add_argument("--arch", default='resnet18', type=str, help='architecture of the target model')
parser.add_argument("--defense", default='no_defense', type=str, help='defense')
parser.add_argument('--noise', type=float, default=0., help='noise std')
parser.add_argument('--p', type=float, default=0., help='p for prun or soteria or adaptive noise')
parser.add_argument('--fi', type=int, default=40, help='fi for adaptive noise')
#parser.add_argument("--normalize", default=False, type=bool, help='normalize or not')


args = parser.parse_args()


#act = 'relu'
act = args.act

size_batch = args.bs

config['device'] = f'cuda:{args.gpu}'
config['classes'] = 10
config['dst'] = "cifar10"

config['defense'] = args.defense

if config['defense'] == 'noise':
    print(args.noise)
    config['noise_std'] = args.noise
    
elif config['defense'] == 'prun':
    config['prun_p'] = args.p
    
elif config['defense'] == 'soteria':
    config['soteria_p'] = args.p
    
elif config['defense'] == 'adapt-noise':
    config['prun_p'] = args.p
    config['noise_std'] = args.noise
    config['fi'] = args.fi
    
else:
    pass
    

if 'LoRA' in args.arch:
    config['LoRA'] = True
    
else:
    config['LoRA'] = False
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

exps = 64 // size_batch

rec_images = []

for exp_i in range(exps):
    if args.arch == 'resnet18':
        '''model = models.resnet18(pretrained=False)
        model.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(in_features=512, out_features=10, bias=True)'''
        model = train_model(num_classes=10, pretrained=False)
        
    elif args.arch == 'vit-base':
        model = CustomVitTimm(scale='base', img_size=32, num_classes=10, pretrained=False)
        config['patch'] = args.patch
        #model.reset_classifier(num_classes)
        #model.eval()
        config['grad_idx'] = args.grad_idx
    elif args.arch == 'vit-tiny':
        model = CustomVitTimm(scale='tiny', img_size=32, num_classes=10, pretrained=False)
        config['patch'] = args.patch
        config['grad_idx'] = args.grad_idx
        #model.eval()
    elif args.arch == 'vit-large-LoRA':
        model =  SimpleViT_LoRA(
            image_size=32,
            patch_size=16,
            num_classes=10,
            dim=1024,
            depth=24,
            heads=16,
            mlp_dim=4096
        )
        model.freeze()
    elif args.arch == 'vit-base-LoRA':
        model =  SimpleViT_LoRA(
            image_size=32,
            patch_size=16,
            num_classes=10,
            dim=768,
            depth=12,
            heads=12,
            mlp_dim=3072
        )
        model.freeze()
    elif args.arch == 'vit-tiny-LoRA':
        model =  SimpleViT_LoRA(
            image_size=32,
            patch_size=16,
            num_classes=10,
            dim=192,
            depth=12,
            heads=3,
            mlp_dim=768
        )
        model.freeze()
        

    #model = models.resnet18(pretrained=False)

    #model.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
    #model.maxpool = nn.Identity()
    #model.fc = nn.Linear(in_features=512, out_features=10, bias=True)

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

    config['rep_freq'] = config['num_epochs']/100

    attack =  {'method':'CI-Net','lr':0.001}
    config['lr']=attack['lr']
    trainer_ = trainer(config,attack,model,dataset, used_idx[size_batch*exp_i:size_batch*(exp_i+1)])        

    reconstructed_image,avg_score = trainer_.attack_training()


    lpips_scorer = lpips.LPIPS(net="alex")

    x_rec = reconstructed_image["0"]["image"]['CI-Net']['ssim']["timeline"][0][-1]

    order = compute_batch_order(lpips_scorer, x_rec, x[size_batch * exp_i:size_batch * (exp_i + 1)])
    x_rec = x_rec[order]

    #for idx in range(x_rec.shape[0]):
        #x_change = 

    std = torch.tensor(cifar10_std).view(3, 1, 1)

    test_psnr = metrics.psnr(x_rec, x[size_batch*exp_i:size_batch*(exp_i+1)], factor=1 / std)
    test_ssim = metrics.cw_ssim(x_rec, x[size_batch*exp_i:size_batch*(exp_i+1)], scales=5)
    lpips_score = lpips_scorer(x_rec, x[size_batch*exp_i:size_batch*(exp_i+1)], normalize=True)

    print(test_psnr)
    print(test_ssim)
    print(torch.mean(lpips_score))
    
    psnrs.append(torch.tensor(test_psnr, dtype=torch.float))
    ssims.append(torch.tensor(test_ssim, dtype=torch.float))
    lpipss.append(torch.mean(lpips_score))

    dm = torch.as_tensor(cifar10_mean)[:, None, None]
    ds = torch.as_tensor(cifar10_std)[:, None, None]

    print(ds.shape)

    #if x_rec.min() < -0.3:
    x_rec = x_rec.mul_(torch.as_tensor(ds)).add_(torch.as_tensor(dm))


    tensor = x_rec.clone().detach().cpu()

    rec_images.append(tensor)


    # 创建保存图像的目录
    #output_dir = f'output_images/cifar10_bs{size_batch}_' + act +f'/exp_{exp_i}'
    #os.makedirs(output_dir, exist_ok=True)

    # 定义一个转换器，将tensor转为PIL Image
    #to_pil_image = transforms.ToPILImage()

    # 迭代地保存每个图像
    '''for i in range(tensor.size(0)):
        img_tensor = tensor[i]

        # 将每个图像的像素值缩放到 0-1 范围
        #img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())

        # 转换为 PIL 图像
        img = to_pil_image(img_tensor)

        # 保存图像为PNG格式
        img.save(os.path.join(output_dir, f'image_{i+1}.png'))

    print(f"Images saved to directory: {output_dir}")'''
#output_all = torch.stack(rec_images)
output_all = torch.cat(rec_images, dim=0)

print(output_all.shape)

grid_plot(output_all, args, ds, dm)
    
psnrs = torch.stack(psnrs)
ssims = torch.stack(ssims)
lpipss = torch.stack(lpipss)

print(f'PSNR: {torch.mean(psnrs)} +- {torch.std(psnrs)}')
print(f'SSIM: {torch.mean(ssims)} +- {torch.std(ssims)}')
print(f'lpips: {torch.mean(lpipss)} +- {torch.std(lpipss)}')