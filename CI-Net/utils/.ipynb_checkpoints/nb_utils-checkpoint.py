# import os
# import io
# import json
# import math
# import threading

import matplotlib.pyplot as plt
# import IPython
# import numpy as np
import torch
# import torchvision.utils as vutils
# from PIL import Image
# from scipy.stats import truncnorm

from piq import ssim,fsim
import time

# -------------- using this function to show the images --------------------------
def plot_img(imgs,name='none'):
    if (imgs.shape[0]==1): #Display when there is only one image.
        fig = plt.figure(figsize=(14,4))
        plt.imshow(torch.clamp(imgs[0],min = 0 ,max =1 ).detach().cpu().permute(1,2,0))
    else:                  #Display when there are many images
        fig, axes = plt.subplots(1, imgs.shape[0], figsize=(imgs.shape[0]*3,5))
        for i,im in enumerate(imgs):
            axes[i].imshow(torch.clamp(im,min = 0 ,max =1 ).detach().cpu().permute(1,2,0).cpu());
            axes[i].axis('off')
    plt.box(False)
    plt.axis('off')
    plt.show()

def calculate_iqa(dataloader,reconstruct_data,config):
    """
        Definition. Calculate the image quality assesment by SSIM and FSIM
        1) Find the pair between original images and reconstructed image based on SSIM and FSIM
            1.1) Each reconstructed image, We find the index of original image that has the highest image quality assesment value
        2) Calculate the averaging IQA values based on the existing pair "pair"
        3) Return the list of averaging IQA value
    """
    start_time = time.time()
    score = {}
    # Calculate the pair based on the SSIM or FSIM
    for metric in ['ssim','fsim']:
        similar_pairs = []
        each_score= 0
        #---------- Finding the pair between ground-truth and reconstructed images -------
        for recon_idx in range(len(reconstruct_data[-1])):      # Loop each reconstructed image 
            max_score = -99
            max_idx = -1
            for ori_idx in range(len(dataloader)):              # Find the ground truth images that are similar with the reconstructed images['recon_idx'] (max SSIM and FSIM)
                if (metric == 'ssim'):
                    each_score = ssim(torch.clamp(torch.stack([dataloader[ori_idx]]).cpu(),min=0,max=1),torch.clamp(torch.stack([reconstruct_data[-1][recon_idx]]),min=0,max=1),data_range=1).cpu().item()
                elif (metric == 'fsim'):
                    each_score = fsim(torch.clamp(torch.stack([dataloader[ori_idx]]).cpu(),min=0,max=1),torch.clamp(torch.stack([reconstruct_data[-1][recon_idx]]),min=0,max=1),data_range=1).cpu().item()
                if (each_score > max_score):                    # Found the better similar image
                    max_score = each_score
                    max_idx = ori_idx
            similar_pairs.append((recon_idx,max_idx)) #append the most similar images together

        #----------- Calculatin the IQA of reconstructed image at each attack round
        score_each_round = []
        each_score = 0
        dataloader = dataloader.to(config['device'])
        #----------- Looping on each attack iteration ---> Calculate the mean SSIM and FSIM at each iteration for plotting the convergence speed
        for attack_iter in range(len(reconstruct_data)):                                            # Loop each attack iteration
            sum_attack_score = 0
            for reconstrcuted_idx, ground_truth_idx in similar_pairs:                               # Loop similar pairs
                reconstruct_data[attack_iter]= reconstruct_data[attack_iter].to(config['device'])
                if (metric == 'ssim'):
                    each_score = ssim(torch.clamp(torch.stack([dataloader[ground_truth_idx]]),min=0,max=1),torch.clamp(torch.stack([reconstruct_data[attack_iter][reconstrcuted_idx]]),min=0,max=1),data_range=1).cpu().item()
                elif (metric == 'fsim'):
                    each_score = fsim(torch.clamp(torch.stack([dataloader[ground_truth_idx]]),min=0,max=1),torch.clamp(torch.stack([reconstruct_data[attack_iter][reconstrcuted_idx]]),min=0,max=1),data_range=1).cpu().item()
                reconstruct_data[attack_iter] = reconstruct_data[attack_iter].to('cpu')
                
                sum_attack_score += each_score
            avg_attack_score = sum_attack_score/ len(reconstruct_data[attack_iter])                 # Average the score from all images
            score_each_round.append(avg_attack_score)
        score[metric] = {'score':score_each_round}
        
    print("Total time for calculating IQA:",(time.time()-start_time)/60)
    return score


# def load_jsonl(logfile):
#     with open(logfile) as f:
#         return [json.loads(x.strip()) for x in f]


# def get_logs(log_dir):
#     return {exp: load_jsonl(os.path.join(log_dir, exp))
#             for exp in os.listdir('logs')
#             if exp.endswith('.jsonl')}


# def plot_log(name, log):

#     itrs = [x['itr'] for x in log]
#     IS_scores = [x['IS_mean'] for x in log]
#     plt.plot(itrs, IS_scores, label=name)

#     plt.legend(loc='lower right', fontsize='x-small')
#     plt.xlabel('Iteration', fontsize='x-large')
#     plt.ylabel('Inception Score', fontsize='x-large')
#     plt.title('Training History', fontsize='xx-large')
#     plt.show()


# def plot_logs(logs):
#     for name, log in logs.items():
#         name = '_'.join(name.split('_'))
#         itrs = [x['itr'] for x in log]
#         IS_scores = [x['IS_mean'] for x in log]
#         plt.plot(itrs, IS_scores, label=name)

#     plt.legend(loc='lower right', fontsize='x-small')
#     plt.xlabel('Iteration', fontsize='x-large')
#     plt.ylabel('Inception Score', fontsize='x-large')
#     plt.title('Training History', fontsize='xx-large')
#     plt.show()


# def smooth_data(data, amount=1.0):
#     if not amount > 0.0:
#         return data
#     data_len = len(data)
#     ksize = int(amount * (data_len // 2))
#     kernel = np.ones(ksize) / ksize
#     return np.convolve(data, kernel, mode='same')


# def parse_log(logfile):
#     seen = {}
#     with open(logfile) as f:
#         for x in f:
#             itr, val = x.strip().split(': ')
#             if itr not in seen:
#                 seen[itr] = val
#         values = seen.values()
#     return list(map(float, values))


# def _parse_log(logfile):
#     with open(logfile) as f:
#         values = [x.strip().split(': ')[1] for x in f]
#     return list(map(float, values))


# def load_logs(log_dir):
#     log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
#     sv_logs = {f: parse_log(os.path.join(log_dir, f))
#                for f in log_files if 'sv0.log' in f}
#     loss_logs = {f: parse_log(os.path.join(log_dir, f))
#                  for f in log_files if 'loss' in f}
#     return {'loss': loss_logs, 'sv': sv_logs}


# def plot_loss_logs(logs, smoothing=0.01, figsize=(15, 15)):
#     G_loss = logs['G_loss.log']
#     D_loss = [x + y for x, y in zip(logs['D_loss_real.log'], logs['D_loss_fake.log'])]
#     G_loss = smooth_data(G_loss, amount=smoothing)
#     D_loss = smooth_data(D_loss, amount=smoothing)
#     plt.figure(figsize=figsize)
#     plt.plot(range(len(G_loss)), G_loss, label='G_loss')
#     plt.plot(range(len(D_loss)), D_loss, label='D_loss')
#     plt.legend(loc='lower right', fontsize='medium')
#     plt.xlabel('Iteration', fontsize='x-large')
#     plt.ylabel('Losses', fontsize='x-large')
#     plt.title('Training History', fontsize='xx-large')
#     # plt.gca().set_ylim(top=10, bottom=-10)
#     plt.gca().set_ylim(top=10, bottom=-10)
#     plt.show()


# def plot_sv_logs(logs):
#     fig, axs = plt.subplots(2)
#     plt.title('Training History', fontsize='xx-large')
#     for name, log in logs.items():
#         itrs = [i * 10 for i in range(len(log))]
#         idx = 0 if name[0] == 'G' else 1
#         axs[idx].plot(itrs, log)

#     for label, ax in zip([r'G $\sigma_0$', r'D $\sigma_0$'], axs.flat):
#         ax.set(ylabel=label)
#     plt.xlabel('Iteration', fontsize='x-large')

#     plt.show()


# def plot_sv_logs(logs, figsize=(15, 15)):
#     fig, axs = plt.subplots(1, 2, figsize=figsize)
# #     plt.title('Training History', fontsize='xx-large')
#     for name, log in logs.items():
#         itrs = [i * 10 for i in range(len(log))]
#         idx = 0 if name[0] == 'G' else 1
#         axs[idx].plot(itrs, log)

#     for label, ax in zip([r'G $\sigma_0$', r'D $\sigma_0$'], axs.flat):
#         ax.set(ylabel=label, xlabel="Iteration")
# #     plt.xlabel('Iteration', fontsize='x-large')

#     plt.show()


# def plot_truncation_curves(trunc_file):
#     with open(trunc_file) as f:
#         coords = [list(map(float, line.strip().split(' '))) for line in f]
#         x, y = list(zip(*coords))
#         plt.plot(x, y, '-')
#         plt.gca().invert_yaxis()
#         plt.xlabel('Inception Score')
#         plt.ylabel('FID')


# def print_stats(logs, blacklist=[]):
#     for name, log in logs.items():
#         #         if name in blacklist:
#         #             continue
#         print_name = '_'.join(name.split('_')[:16])
#         if print_name in blacklist:
#             continue
#         max_IS_idx = np.argmax([x['IS_mean'] for x in log])
#         min_FID_idx = np.argmin([x['FID'] for x in log])
#         last_itr = log[-1]['itr']

#         max_IS = log[max_IS_idx]['IS_mean']
#         min_FID = log[min_FID_idx]['FID']
#         max_IS_itr = log[max_IS_idx]['itr']
#         min_FID_itr = log[min_FID_idx]['itr']
#         print(f'{print_name}\n'
#               f'\t current itr: {last_itr}\n'
#               f'\t current IS: {log[-1]["IS_mean"]:.3f}\n'
#               f'\t current FID: {log[-1]["FID"]:.3f}\n'
#               f'\t max IS: {max_IS:.3f} at itr ({max_IS_itr})\n'
#               f'\t min FID: {min_FID:.3f} at itr ({min_FID_itr})')


# def plot_IS_FID(logs):
#     fig, axs = plt.subplots(2, sharex=True)
#     for name, log in logs.items():
#         name = '_'.join(name.split('_'))
#         itrs = [x['itr'] for x in log]
#         IS_scores = [x['IS_mean'] for x in log]
#         FID = [x['FID'] for x in log]
#         axs[0].plot(itrs, IS_scores, label=name)
#         axs[1].semilogy(itrs, FID, label=name)

#     for label, ax in zip(['Inception Score', 'FID'], axs.flat):
#         ax.set(ylabel=label)

#     plt.xlabel('Iteration', fontsize='x-large')
#     plt.legend(loc='upper right', fontsize='x-small')
#     axs[0].set_title('Training History', fontsize='xx-large')
#     fig.tight_layout()
#     plt.show()


# def visualize_data(data, num_samples=64, figsize=(15, 15), title='Real Images'):
#     if isinstance(data, torch.utils.data.Dataset):
#         print(data)
#         samples = torch.stack([data[i][0] for i in range(num_samples)])
#     elif isinstance(data, torch.utils.data.DataLoader):
#         print(data.dataset)
#         samples = next(iter(data))[0][:num_samples]
#     else:
#         raise ValueError(f'Unrecognized data source type: {type(data)}'
#                          'Must be instance of either torch Dataset or DataLoader')
#     visualize_samples(samples, figsize=figsize, title=title)


# def visualize_samples(samples, figsize=(15, 15), title='Samples',
#                       nrow=8, padding=5, normalize=True, scale_each=False, use_plt=False):
#     # Plot the real images
#     im = vutils.make_grid(samples, nrow=nrow, padding=padding,
#                           normalize=normalize, scale_each=scale_each).cpu()
#     if use_plt:
#         plt.figure(figsize=figsize)
#         plt.axis("off")
#         plt.title(title)
#         plt.imshow(np.transpose(im, (1, 2, 0)))
#     else:
#         imshow(np.transpose(255 * im, (1, 2, 0)))


# def imshow(image, format='png', jpeg_fallback=True):
#     image = np.asarray(image, dtype=np.uint8)
#     str_file = io.BytesIO()
#     Image.fromarray(image).save(str_file, format)
#     im_data = str_file.getvalue()
#     try:
#         disp = IPython.display.display(IPython.display.Image(im_data))
#     except IOError:
#         if jpeg_fallback and format != 'jpeg':
#             print('Warning: image was too large to display in format "{}"; '
#                   'trying jpeg instead.').format(format)
#             return imshow(image, format='jpeg')
#         else:
#             raise
#     return disp


# def smooth_data(data, amount=1.0):
#     if not amount > 0.0:
#         return data
#     data_len = len(data)
#     ksize = int(amount * (data_len // 2))
#     kernel = np.ones(ksize) / ksize
#     return np.convolve(data, kernel, mode='same')


# def _save_sample(G, fixed_noise, filename, nrow=8, padding=2, normalize=True):
#     fake_image = G(fixed_noise).detach()
#     vutils.save_image(fake_image, filename, nrow=nrow, padding=padding, normalize=normalize)


# def save_samples(G, fixed_noise, filename, threaded=True):
#     if threaded:
#         G.to('cpu')
#         thread = threading.Thread(name='save_samples',
#                                   target=_save_sample,
#                                   args=(G, fixed_noise, filename))
#         thread.start()
#     else:
#         _save_sample(G, fixed_noise, filename)


# def slerp(start, end, weight):
#     """TODO: Finish."""
#     low_norm = start / torch.norm(start, dim=1, keepdim=True)
#     high_norm = end / torch.norm(end, dim=1, keepdim=True)
#     omega = torch.acos((low_norm * high_norm).sum(-1))
#     so = torch.sin(omega)
#     print('ip', (low_norm * high_norm).sum(-1).shape)
#     print(f'low_norm: {low_norm.shape}')
#     print(f'high_norm: {high_norm.shape}')
#     print(f'omega: {omega.shape}')
#     print(f'so: {so.shape}')
#     # print(f'weight: {weight.shape}')
#     print((weight * omega / so).shape)
#     # res = ((torch.sin((1.0 - weight) * omega) / so).unsqueeze(1) * start
#     #    + (torch.sin(weight * omega) / so).unsqueeze(1) * end)
#     res = ((torch.sin((1.0 - weight) * omega) / so) * start
#            + (torch.sin(weight * omega) / so) * end)
#     return res


# def interp(x0, x1, num_midpoints, device='cuda', interp_func=torch.lerp):
#     """Interpolate between x0 and x1.

#     Args:
#         x0 (array-like): Starting coord with shape [batch_size, ...]
#         x1 (array-like): Ending coord with shape [batch_size, ...]
#         num_midpoints (int): Number of midpoints to interpolate.
#         device (str, optional): Device to create interp. Defaults to 'cuda'.
#     """
#     x0 = x0.view(x0.size(0), 1, *x0.shape[1:])
#     x1 = x1.view(x1.size(0), 1, *x1.shape[1:])
#     lerp = torch.linspace(0, 1.0, num_midpoints + 2, device=device).to(x0.dtype)
#     lerp = lerp.view(1, -1, 1)
#     return interp_func(x0, x1, lerp)


# def truncated_z_sample(batch_size, dim_z, truncation=1.0, seed=None, device='cuda'):
#     state = None if seed is None else np.random.RandomState(seed)
#     values = truncnorm.rvs(-2, 2, size=(batch_size, dim_z), random_state=state)
#     return torch.Tensor(float(truncation) * values).to(device)


# def make_grid(tensor, nrow=8):
#     """Make a grid of images."""
#     tensor = np.array(tensor)
#     nmaps = tensor.shape[0]
#     xmaps = min(nrow, nmaps)
#     ymaps = int(math.ceil(float(nmaps) / xmaps))
#     height, width = int(tensor.shape[1]), int(tensor.shape[2])
#     grid = np.zeros((height * ymaps, width * xmaps, 3), dtype=np.uint8)
#     k = 0
#     for y in range(ymaps):
#         for x in range(xmaps):
#             if k >= nmaps:
#                 break
#             grid[y * height: (y + 1) * height,
#                  x * width: (x + 1) * width] = tensor[k]
#             k = k + 1
#     return grid

