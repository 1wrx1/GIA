import torch
import torch.nn as nn
import layers
import torch.nn.functional as F
import torch.optim as optim


from math import sqrt
from tqdm.notebook import trange
from utils.reconstructed import loss_inverting_gt
from utils.nb_utils import calculate_iqa

class trainer():
    """
        The attack process only involves a training process
    """
    def __init__ (self,config,attack_config,model,dataset):
        self.config = config
        self.attack = attack_config
        self.model = model
        self.dataset = dataset

    def attack_training(self):
        avg_score = {}
        reconstructed_image = {}
        all_dataloader={}
        used_idx=[]
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

            #-- Generate the ground-truth batch (images and labels)--#
            for i in range(self.config['total_img']):
                idx = torch.arange(0,len(self.dataset))[i]             # pick the first few images
                # idx = torch.randint(0,len(dataset),(1,))        # pick the image (idx) randomly
                # while idx in used_idx:                          # Keep pick the new images when it has already been choosen
                # idx = torch.randint(0,len(dataset),(1,))
                batch_img.append(self.dataset[idx][0])
                batch_label.append(self.dataset[idx][1])
                used_idx.append(idx.item())

            dataloader = torch.stack(batch_img).to(self.config['device'])
            gt_label = torch.as_tensor(batch_label).to(self.config['device'])      
            all_dataloader[str(num_exp)]=dataloader     # Save the ground-truth image
            self.config['data_shape'] = dataloader.size()    # Retrived the data shape to the configuration

            # Calculate the ground-truth gradients (shared gradients from participant)
            output = self.model(dataloader)                                  # Predicted the output
            criterion = nn.CrossEntropyLoss().to(self.config['device'])      # Create the loss function
            loss = criterion(output,gt_label)                           # Calculate the loss (Assuming that we extract the label)
            dy_dx = torch.autograd.grad(loss, self.model.parameters())       # Compute dy_dx
            original_dy_dx = list((_.detach().clone() for _ in dy_dx))  
            
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
        if self.config["dst"]=="cifar10":
            self.img_res = 32
        elif self.config["dst"]=="imagenet":
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
        model           = model.to(device)
    
        print('Configuration Parameters :',self.config)

        channel = self.over_parameterization()
        print("The channel number is {}".format(channel))
        # Define Generator and noise        
        self.netG = Generator(image_res=self.img_res,in_channel=channel).to(device)
        self.noise = torch.randn(b_size,nz, device=device) 
        # y = torch.randint(10, (b_size,), device=device).long()  ###ZC:ImageNet
        optimizerG = optim.Adam(self.netG.parameters(), lr=lr)
        
        if lr_decay:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizerG,milestones=[num_epochs // 2.667, num_epochs// 1.6,num_epochs // 1.142], gamma=0.1)   # 3/8 5/8 7/8

        # Lists to keep track of progress
        G_losses = []
        image_recon = []

        # Start the reconstrcution attack
        # with trange(num_epochs,disable=True) as t:
            # for iters in (t):
        for iters in trange(num_epochs, desc="Processing"):
            # print("Iteration {}".format(iters))                  
            optimizerG.zero_grad()

            fake  = self.netG(self.noise).to(self.config['device'])
            #Passing the fake input to the global model
            fake_output = model(fake)
            criterion = nn.CrossEntropyLoss().to(device)
            #Calculating the dummy gradient
            dummy_loss = criterion(fake_output,original_label)
            fake_dy_dx = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)
            #Calculating the loss between the original gradients and dummy gradients
            errG = loss_inverting_gt(original_gt,fake_dy_dx,fake,tv_value)
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
            self.to_rgb_64 = nn.Conv2d(in_channel, 3, 1)
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
        
        
        