import matplotlib.pyplot as plt
import numpy as np
import os 


def grid_plot(tensor, bs, args):
    tensor = tensor.clone().detach()
    num_images = 64

    fig, axes = plt.subplots(num_images // 8, 8, figsize=(12, num_images // 16 * 3))
    axes = np.reshape(axes, -1)
    for im, ax in zip(tensor, axes):
        ax.imshow(im.permute(1, 2, 0).cpu())
        ax.axis('off')
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    os.makedirs('saved_img') if not os.path.exists('saved_img') else None

    if args.pretrain:
        file_name = f'saved_img/ImageNet_{args.resolution}_bs_{bs}_' + args.arch + '_pretrain.pdf'
    else:
        file_name = f'saved_img/ImageNet_{args.resolution}_bs_{bs}_' + args.arch + '_unpretrain.pdf'

    plt.savefig(file_name)