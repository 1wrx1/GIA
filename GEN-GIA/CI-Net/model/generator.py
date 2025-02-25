import torch
import torch.nn as nn
import layers
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T


from math import sqrt
from tqdm.notebook import trange
from utils.reconstructed import loss_inverting_gt
from utils.nb_utils import calculate_iqa

import numpy as np

from transformers import (
    CLIPProcessor,
    CLIPModel,
    get_scheduler
)
from PIL import Image
from peft import get_peft_model, LoraConfig


def compute_risk(model: nn.Module):
    """Compute risks."""
    var = []
    for param in model.parameters():
        var.append(torch.var(param).cpu().detach().numpy())
    var = [min(v, 1) for v in var]
    return var
    

class trainer():
    """
        The attack process only involves a training process
    """
    def __init__ (self,config,attack_config,model,dataset, used_idx):
        self.config = config
        self.attack = attack_config
        self.model = model
        self.dataset = dataset
        self.used_idx = used_idx

    def attack_training(self):
        avg_score = {}
        reconstructed_image = {}
        all_dataloader={}
        #used_idx=[]
        #  Conduct the attacks 10 times with different images
        for num_exp in range(1):  #change to 1
            batch_img = []
            batch_label = []

            # Assign parameters for saving the experimental results
            experimental_results = {}
            reconstructed_image[str(num_exp)]={}
            reconstructed_image[str(num_exp)]['config']={}
            reconstructed_image[str(num_exp)]['image']={}
            avg_score[str(num_exp)] = {}

            if self.config['CLIP']:
                pass
            else:
                for idx in self.used_idx:
                    print(idx)
                    batch_img.append(self.dataset[idx][0])
                    batch_label.append(self.dataset[idx][1])
    
                dataloader = torch.stack(batch_img).to(self.config['device'])
                gt_label = torch.as_tensor(batch_label).to(self.config['device'])      
                all_dataloader[str(num_exp)]=dataloader     # Save the ground-truth image
                self.config['data_shape'] = dataloader.size()    # Retrived the data shape to the configuration
    
                # Calculate the ground-truth gradients (shared gradients from participant)
                output = self.model(dataloader)                                  # Predicted the output
                
                if self.config['defense'] == 'soteria':
                    soteria_p = self.config['soteria_p'] * 100
                    
                    dataloader.requires_grad = True
                    
                    feature_fc1_graph = self.model.get_feature(dataloader)
                    deviation_f1_target = torch.zeros_like(feature_fc1_graph)
                    deviation_f1_x_norm = torch.zeros_like(feature_fc1_graph)
                    for f in range(deviation_f1_x_norm.size(1)):
                        deviation_f1_target[:,f] = 1
                        feature_fc1_graph.backward(deviation_f1_target, retain_graph=True)
                        deviation_f1_x = dataloader.grad.data
                        deviation_f1_x_norm[:,f] = torch.norm(deviation_f1_x.view(deviation_f1_x.size(0), -1), dim=1)/(feature_fc1_graph.data[:,f] + 0.1)
                        self.model.zero_grad()
                        dataloader.grad.data.zero_()
                        deviation_f1_target[:,f] = 0
                    
                    deviation_f1_x_norm_sum = deviation_f1_x_norm.sum(axis=0)
                    thresh = np.percentile(deviation_f1_x_norm_sum.flatten().cpu().numpy(), soteria_p)
                    mask = np.where(abs(deviation_f1_x_norm_sum.cpu()) < thresh, 0, 1).astype(np.float32)
                
                criterion = nn.CrossEntropyLoss().to(self.config['device'])      # Create the loss function
                loss = criterion(output,gt_label)                           # Calculate the loss (Assuming that we extract the label)
                
                if self.config['LoRA']:
                    params_with_grad = [p for p in self.model.parameters() if p.requires_grad]
                    #input_gradient = torch.autograd.grad(target_loss, params_with_grad)
                    dy_dx = torch.autograd.grad(loss, params_with_grad)       # Compute dy_dx
                else:
                    dy_dx = torch.autograd.grad(loss, self.model.parameters())       # Compute dy_dx
                    
                original_real_dy_dx = list((_.detach().clone() for _ in dy_dx))
                
                if self.config['defense'] == 'no_defense':
                    original_dy_dx = original_real_dy_dx
                    
                elif self.config['defense'] == 'noise':
                    noise_var = self.config['noise_std']
    
                    print(noise_var)
                    
                    original_dy_dx = [grad + torch.normal(torch.zeros_like(grad), noise_var * torch.ones_like(grad)) for grad in original_real_dy_dx]
                    print('Noise')
                    
                elif self.config['defense'] == 'prun':
                    percentage = self.config['prun_p'] * 100
                    device = original_real_dy_dx[0].device
                    original_dy_dx = [None] * len(original_real_dy_dx)
                    for i in range(len(original_real_dy_dx)):
                        grad_tensor = original_real_dy_dx[i].clone().cpu().numpy()
                        flattened_weights = np.abs(grad_tensor.flatten())
                        thresh = np.percentile(flattened_weights, percentage)
                        grad_tensor = np.where(abs(grad_tensor) < thresh, 0, grad_tensor)
                        original_dy_dx[i] = torch.Tensor(grad_tensor).to(device)
                    print('Prun')
                    
                    
                
                elif self.config['defense'] == 'soteria':
                    #percentage = self.config['soteria_p'] * 100
                    device = original_real_dy_dx[0].device
                    
                    original_dy_dx = original_real_dy_dx
                    
                    gradient_fc = original_real_dy_dx[-2]
                    gradient_fc = gradient_fc * torch.Tensor(mask).to(device)
                    
                    original_dy_dx[-2] = gradient_fc
                    
                    print('Soteria')
    
                elif self.config['defense'] == 'no_fc':
                    original_dy_dx = original_real_dy_dx[:-2]
    
                    print('no_fc')
                    
                elif self.config['defense'] == 'adapt-noise':
                    device = original_real_dy_dx[0].device
                    
                    risk = compute_risk(self.model)
                    # Calculate empirical FIM
                    fim = []
                    flattened_fim = None
                    
                    original_dy_dx = [None] * len(original_real_dy_dx)
                    
                    for _, grad in enumerate(original_real_dy_dx):
                        squared_grad = grad.clone().pow(2).mean(0).cpu().numpy()
                        fim.append(squared_grad)
                        if flattened_fim is None:
                            flattened_fim = squared_grad.flatten()
                        else:
                            flattened_fim = np.append(flattened_fim, squared_grad.flatten())
                
                    fim_thresh = np.percentile(flattened_fim, (100 - self.config['fi'])) #fi set to 40 (for now)
                
                    for i, grad in enumerate(original_real_dy_dx):
                        # pruning
                        grad_tensor = grad.cpu().numpy()
                        flattened_weights = np.abs(grad_tensor.flatten())
                        thresh = np.percentile(flattened_weights, 100 * self.config['prun_p'])
                        grad_tensor = np.where(abs(grad_tensor) < thresh, 0, grad_tensor)
                
                        # noise
                        noise_base = torch.normal(
                            0, risk[i] * self.config['noise_std'], grad.shape
                        )
                        noise_mask = np.where(fim[i] < fim_thresh, 0, 1)
                        gauss_noise = noise_base * noise_mask
                        grad = (
                            (torch.Tensor(grad_tensor) + gauss_noise)
                            .to(dtype=torch.float32)
                            .to(device)
                        )
                        
                        original_dy_dx[i] = grad
            
            
            print('-----------------ATTACK BEGIN----------------')
            attack_worker = CI_attacker(self.config)
            experimental_results[self.attack['method']] = attack_worker.reconstructed_gt(original_dy_dx,\
                                                    gt_label,self.model)         # Conduct the attack by minimizing the loss between dummy gradients and original gradients
            print('-----------------ATTACK END----------------')

            #-- Compute Image Quality --#
            avg_score[str(num_exp)][self.attack['method']] = calculate_iqa(dataloader,experimental_results[self.attack['method']],self.config)    # Calculate the iqa score
            reconstructed_image[str(num_exp)]['image'][self.attack['method']] ={}  
            for iqa in ['ssim','fsim']: 
                reconstructed_image[str(num_exp)]['image'][self.attack['method']][iqa] = {
                'timeline' :[experimental_results[self.attack['method']]]                                                                                    # Take a snap shot of reconstruction attack 
                    }
                # In some case  that the higest score is not the last attack iteration, we will save the highest iqa index and image
                highest_iqa_at_idx = avg_score[str(num_exp)][self.attack['method']][iqa]['score'].index(max(avg_score[str(num_exp)][self.attack['method']][iqa]['score']))
                if highest_iqa_at_idx == self.config['num_epochs']-1:  
                    reconstructed_image[str(num_exp)]['image'][self.attack['method']][iqa]['highest_score_idx'] = -1                                             # The last reconstructed images are the highest IQA
                else:
                    print('Hightest IQA at other index',highest_iqa_at_idx)
                    reconstructed_image[str(num_exp)]['image'][self.attack['method']][iqa]['highest_score_idx'] = highest_iqa_at_idx     ## The last reconstructed images are not the highest IQA
                    reconstructed_image[str(num_exp)]['image'][self.attack['method']][iqa]['highest_score_img'] =  experimental_results[self.attack['method']][highest_iqa_at_idx]
                # experimental_results[attack['method']] = []
            print('-----------------IMAGE ASSESSMENT END----------------')
        return reconstructed_image,avg_score


class CI_attacker():
    """
        Return the attacker object for reconstruction attack based on the hyper parameter setting
        .reconstructed_gt(): It will reconstruct the input images based on the updated gradients.
        
    """
    def __init__ (self,config):
        self.config = config
        if self.config["dst"] in ["cifar10", "cifar100", 'celeba32']:
            self.img_res = 32
        elif self.config['dst'] in ['celeba64', 'imagenet64']:
            self.img_res = 64
        elif self.config["dst"]=="imagenet":
            self.img_res = 128
        elif self.config['dst']=='imagenet256':
            self.img_res = 256
        else:
            print("Please set the image resolution manually")

    def over_parameterization(self):
        """ This function sets the channel number of GI_Net,
          so that the generator is over-parameterized
        """
        batch_size = self.config["b_size"]
        
        image_params = batch_size * (self.img_res**2)
        channel = 32
        model_params = sum(p.numel() for p in Generator(in_channel=channel).parameters())
        while model_params <= 4*image_params:
            channel = channel*2
            model_params = sum(p.numel() for p in Generator(in_channel=channel).parameters())
        return channel


    def reconstructed_gt(self,original_gt,original_label,model):

        # Set the configuration based on variable "config()"
        device          = self.config['device']
        nz              = self.config['nz']
        b_size          = self.config['b_size']
        lr              = self.config['lr']
        tv_value        = self.config['tv_value']
        num_epochs      = self.config['num_epochs']
        lr_decay        = self.config['lr_decay']
        rep_freq        = self.config['rep_freq']
        patch           = self.config['patch']
        idx             = self.config['grad_idx']
        model           = model.to(device)
    
        print('Configuration Parameters :',self.config)

        channel = self.over_parameterization()
        print("The channel number is {}".format(channel))
        # Define Generator and noise        
        self.netG = Generator(image_res=self.img_res,in_channel=channel).to(device)
        self.noise = torch.randn(b_size,nz, device=device) 
        optimizerG = optim.Adam(self.netG.parameters(), lr=lr)
        
        if lr_decay:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizerG,milestones=[num_epochs // 2.667, num_epochs// 1.6,num_epochs // 1.142], gamma=0.1)   # 3/8 5/8 7/8

        G_losses = []
        image_recon = []

        for iters in trange(num_epochs, desc="Processing"):             
            optimizerG.zero_grad()


            fake = self.netG(self.noise).to(device)

            if self.config['CLIP']:
                pass

            else:
            
                fake_output = model(fake)
                criterion = nn.CrossEntropyLoss().to(device)
                #Calculating the dummy gradient
                dummy_loss = criterion(fake_output,original_label)
                fake_params_with_grad = [p for p in model.parameters() if p.requires_grad]
                #input_gradient = torch.autograd.grad(target_loss, params_with_grad)
                fake_origin_dy_dx = torch.autograd.grad(dummy_loss, fake_params_with_grad, create_graph=True)
            
            if self.config['defense'] == 'prun':
                gradients = list(fake_origin_dy_dx)
                for i in range(len(gradients)):
                    gradient = gradients[i]
                    shape = gradient.shape
                    # print(gradient.shape)
                    gradient = torch.flatten(gradient)
                    _, indice = torch.topk(torch.abs(gradient), int(self.config['prun_p'] * len(gradient)), largest=False)
                    gradient[indice] = 0
                    gradient = gradient.reshape(shape)
                    gradients[i] = gradient
                fake_dy_dx = tuple(gradients)
                

            elif self.config['defense'] == 'no_fc':
                fake_dy_dx = fake_origin_dy_dx[:-2]

            else:
                fake_dy_dx = fake_origin_dy_dx
            
            #Calculating the loss between the original gradients and dummy gradients
            errG = loss_inverting_gt(original_gt,fake_dy_dx,fake,tv_value, patch)
            errG.backward()
            
            # Changing the value of gradient to sign only. 
            if self.config['signed']:
                for layer in self.netG.parameters():
                    layer.grad.sign_()
            # Update G
            optimizerG.step()
            # Save Losses for plotting later
            G_losses.append(errG.item())
            # Save the reconstructed image on each to variable "image_recon"
            fake = fake.detach().cpu()
    
            if (iters+1)%rep_freq==0 :
                image_recon.append(fake)
            iters += 1
            # t.set_postfix(gt_loss = errG.item()) # for monitoring only 
            if lr_decay:
                scheduler.step()

        return image_recon


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding):
        super().__init__()
        convs = [layers.SNConv2d(in_channel, out_channel, kernel_size, padding=padding)]
        convs.append(nn.BatchNorm2d(out_channel))
        convs.append(nn.LeakyReLU(0.1))
        convs.append(layers.SNConv2d(out_channel, out_channel, kernel_size, padding=padding))
        convs.append(nn.BatchNorm2d(out_channel))
        convs.append(nn.LeakyReLU(0.1))
        self.conv = nn.Sequential(*convs)

    def forward(self, input):
        out = self.conv(input)
        return out    

def upscale(feat):
    return F.interpolate(feat, scale_factor=2) #, mode="bilinear") 


class Generator(nn.Module):
    def __init__(self, image_res=32, input_code_dim=128, in_channel=256, tanh=True):
        super().__init__()
        self.image_res = image_res
        self.input_dim = input_code_dim
        self.tanh = tanh
        self.input_layer = nn.Sequential(
            nn.ConvTranspose2d(input_code_dim, in_channel, 4, 1, 0),
            nn.BatchNorm2d(in_channel),
            nn.LeakyReLU(0.1))

        self.progression_4 = ConvBlock(in_channel, in_channel, 3, 1)#, pixel_norm=pixel_norm) 
        self.progression_8 = ConvBlock(in_channel, in_channel, 3, 1)#, pixel_norm=pixel_norm) 
        self.progression_16 = ConvBlock(in_channel, in_channel, 3, 1)#, pixel_norm=pixel_norm) 
        self.progression_32 = ConvBlock(in_channel, in_channel, 3, 1)#, pixel_norm=pixel_norm)
        if self.image_res >= 64:
            self.progression_64 = ConvBlock(in_channel, in_channel//2, 3, 1) # pixel_norm=pixel_norm)
        if self.image_res >= 128:
            self.progression_128 = ConvBlock(in_channel//2, in_channel//4, 3, 1) #, pixel_norm=pixel_norm)
        if self.image_res >= 256:
            self.progression_256 = ConvBlock(in_channel//4, in_channel//4, 3, 1) #, pixel_norm=pixel_norm)

        if self.image_res == 32:
            self.to_rgb_32 = nn.Conv2d(in_channel, 3, 1)
        if self.image_res == 64:
            self.to_rgb_64 = nn.Conv2d(in_channel//2, 3, 1)
        if self.image_res == 128:
            self.to_rgb_128 = nn.Conv2d(in_channel//4, 3, 1)
        if self.image_res == 256:
            self.to_rgb_256 = nn.Conv2d(in_channel//4, 3, 1)
        
        self.max_step = 6

    def progress(self, feat, module):
        out = F.interpolate(feat, scale_factor=2) #, mode="bilinear")
        out = module(out)
        return out

    def output_simple(self, feat1,  module1, alpha):
        out = module1(feat1)
        if self.tanh:
            return torch.tanh(out)
        return out

    def forward(self, input, step=6, alpha=0):
        if step > self.max_step:
            step = self.max_step

        out_4 = self.input_layer(input.view(-1, self.input_dim, 1, 1))
        out_4 = self.progression_4(out_4)
        out_8 = self.progress(out_4, self.progression_8)        
        out = self.progress(out_8, self.progression_16)

        resolutions = [32, 64, 128, 256]

        for res in resolutions:
            if self.image_res >= res:
                out = self.progress(out, getattr(self, f'progression_{res}'))
                if self.image_res == res:
                    #print(res)
                    return self.output_simple(out, getattr(self, f'to_rgb_{res}'), alpha)


        # out_32 = self.progress(out_16, self.progression_32)
        # if self.image_res == 32:
        #     return self.output_simple(out_32,self.to_rgb_32,alpha)
        # if self.image_res >= 64:
        #     out_64 = self.progress(out_32, self.progression_64)
        #     if self.image_res == 64:
        #         return self.output_simple(out_64,self.to_rgb_64,alpha)
        # if self.image_res >= 128:
        #     out_128 = self.progress(out_64, self.progression_128)
        #     if self.image_res == 128:
        #         return self.output_simple(out_128,self.to_rgb_128,alpha)
        # if self.image_res >= 256:
        #     out_256 = self.progress(out_128, self.progression_256)
        #     if self.image_res == 256:
        #         return self.output_simple(out_256,self.to_rgb_256,alpha)
        
        
        