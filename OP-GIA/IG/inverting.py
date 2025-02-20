import torch
import torchvision
import numpy as np
from argparse import ArgumentParser
import inversefed
import os
import random
import lpips
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import defense
torch.set_num_threads(4)


def normalize_tensor(in_feat, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat ** 2, dim=1, keepdim=True))
    return in_feat / (norm_factor + eps)


def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2, 3], keepdim=keepdim)


def compute_batch_order(lpips_scorer, output, ground_truth, setup):
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
        similarity_matrix = torch.zeros(B, B, **setup)
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
    return torch.as_tensor(rec_assignment, device=setup["device"], dtype=torch.long)


if __name__ == '__main__':
    parser = ArgumentParser("Inverting Gradients")
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument("--arch", default='ResNet18', type=str, help='model architecture')
    parser.add_argument("--trained", default=False, type=bool, help='trained or not')
    parser.add_argument("--dataset", default='CIFAR10', type=str, help='CIFAR10, CIFAR100, ImageNet')
    parser.add_argument("--num_classes", default=10, type=int, help='number of classes')
    parser.add_argument("--file", default='resnet18_CIFAR10_epoch_120.pth', type=str, help='trained model file name')
    parser.add_argument("--img_shape", default=32, type=int, help='image shape')
    parser.add_argument("--num_images", default=64, type=int, help='number of images')
    parser.add_argument("--idx", default=48, type=int, help='select image for bs=1')
    parser.add_argument("--cost_fn", default='sim', type=str, help='distance metric: sim or l2')
    parser.add_argument("--lr", default=0.1, type=float, help='learning rate for recovering algorithm')
    parser.add_argument("--iter", default=24000, type=int, help='number of iterations')
    parser.add_argument("--restarts", default=1, type=int, help='times of recovering')
    parser.add_argument("--batch_size", default=1, type=int, help='batch size')
    # below for attacking FedAvg
    parser.add_argument("--attack_fedavg", default=False, type=bool, help='attack FedSGD or FedAvg')
    parser.add_argument("--epochs", default=1, type=int, help='number of epochs for local training')
    parser.add_argument("--local_lr", default=0.01, type=float, help='learning rate for local training')
    parser.add_argument("--simulation", default=0, type=int, help='0:strong simulation; 1:weak simulation; 2:no simulation')
    # defense
    parser.add_argument('--defense', type=str, default=None, choices=[None, 'compression', 'noise'])
    parser.add_argument('--d_param', type=float, default=None, help='Parameter setting for the defense, i.e., var for noise, and pruning rate for compression.')

    args = parser.parse_args()
    print(args)

    setup = inversefed.utils.system_startup(gpu=args.gpu)
    defs = inversefed.training_strategy('conservative')
    if args.dataset == 'ImageNet':
        data_path = '/mnt/7T/Data/Natural/ImageNet/'
        if args.arch == 'ResNet18':
            model = torchvision.models.resnet18(pretrained=args.trained)
        else:
            raise ('No support arch for ImageNet.')
    else:
        data_path = '/mnt/7T/Data/Natural/'
        model, _ = inversefed.construct_model(args.arch, num_classes=args.num_classes, num_channels=3)
    loss_fn, _, data_loader = inversefed.construct_dataloaders(args.dataset, defs, data_path=data_path)
    model.to(**setup)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    if args.trained:
        if args.dataset == 'ImageNet':
            defs.lr = 0.0001
        args.file = f'{args.arch}_{args.dataset}_epoch_{defs.epochs}.pth'
        os.makedirs('trained_models') if not os.path.exists('trained_models') else None
        try:
            model.load_state_dict(torch.load(f'trained_models/{args.file}'))
        except FileNotFoundError:
            inversefed.train(model, loss_fn, data_loader, data_loader, defs, setup=setup)
            torch.save(model.state_dict(), f'trained_models/{args.file}')
    model.eval()

    # normalize parameter
    if args.dataset == 'CIFAR10':
        dm = torch.as_tensor(inversefed.consts.cifar10_mean, **setup)[:, None, None]
        ds = torch.as_tensor(inversefed.consts.cifar10_std, **setup)[:, None, None]
    elif args.dataset == 'CIFAR100':
        dm = torch.as_tensor(inversefed.consts.cifar100_mean, **setup)[:, None, None]
        ds = torch.as_tensor(inversefed.consts.cifar100_std, **setup)[:, None, None]
    elif args.dataset == 'ImageNet':
        dm = torch.as_tensor(inversefed.consts.imagenet_mean, **setup)[:, None, None]
        ds = torch.as_tensor(inversefed.consts.imagenet_std, **setup)[:, None, None]
    else:
        raise ('No support dataset.')

    # all data needed to be recovered
    ground_truth_all, labels_all = [], []
    idx = args.idx
    while len(labels_all) < args.num_images:
        img, label = data_loader.dataset[idx]
        idx += 666 if args.dataset == 'ImageNet' else 1
        if args.dataset in ['CIFAR100', 'ImageNet']:
            # skip the images with same labels when images <= num_classes
            if label not in labels_all or len(labels_all) >= args.num_classes:
                labels_all.append(torch.as_tensor((label,), device=setup['device']))
                ground_truth_all.append(img.to(**setup))
        else:
            labels_all.append(torch.as_tensor((label,), device=setup['device']))
            ground_truth_all.append(img.to(**setup))

    # different config for ImageNet and CIFAR
    if args.dataset == 'ImageNet':
        config = dict(signed=True, boxed=True, cost_fn=args.cost_fn, indices='top10', weights='equal', lr=args.lr,
                      optim='adam', restarts=args.restarts, max_iterations=args.iter, total_variation=1e-6,
                      init='randn', filter='median', lr_decay=True, scoring_choice='loss')
    else:
        config = dict(signed=True, boxed=True, cost_fn=args.cost_fn, indices='def', weights='equal', lr=args.lr,
                      optim='adam', restarts=args.restarts, max_iterations=args.iter, total_variation=1e-2,
                      init='randn', filter='none', lr_decay=True, scoring_choice='loss')

    if not args.attack_fedavg:
        # divide into batches
        list_all = list(range(args.num_images))
        random.shuffle(list_all)
        list_all = [list_all[i:i + args.batch_size] for i in range(0, len(list_all), args.batch_size)]
        # print(list_all)
        output_all = []
        rec_loss = []
        for lists in list_all:
            ground_truth = torch.stack([ground_truth_all[i] for i in lists])
            labels = torch.cat([labels_all[i] for i in lists])

            model.zero_grad()
            target_loss, _, _ = loss_fn(model(ground_truth), labels)

            input_gradient = torch.autograd.grad(target_loss, model.parameters())
            input_gradient = [grad.detach() for grad in input_gradient]
            full_norm = torch.stack([g.norm() for g in input_gradient]).mean()
            print(f'Full gradient norm is {full_norm:e}.')

            if args.defense is None:
                print('No defense applied.')
            else:
                if args.defense == 'noise':
                    d_param = 0.01 if args.d_param is None else args.d_param
                    input_gradient = defense.additive_noise(input_gradient, var=d_param)
                elif args.defense == 'compression':
                    d_param = 90 if args.d_param is None else args.d_param
                    input_gradient = defense.gradient_compression(input_gradient, percentage=d_param)
                else:
                    raise NotImplementedError("Invalid defense method!")
                print('Defense applied: {} w/ {}.'.format(args.defense, d_param))

            rec_machine = inversefed.GradientReconstructor(model, (dm, ds), config, num_images=args.batch_size)
            output, stats = rec_machine.reconstruct(input_gradient, labels,
                                                    img_shape=(3, args.img_shape, args.img_shape))
            output_all.append(output)
            rec_loss.append(stats['opt'])

        ground_truth_all = torch.stack(ground_truth_all)
        output_all = torch.cat(output_all, dim=0)
    else:
        ground_truth = torch.stack(ground_truth_all)
        labels = torch.cat(labels_all)

        model.zero_grad()
        target_loss, _, _ = loss_fn(model(ground_truth), labels)

        local_steps = args.num_images // args.batch_size * args.epochs
        input_parameters = inversefed.reconstruction_algorithms.loss_steps(model, ground_truth, labels,
                                                                           lr=args.local_lr, local_steps=local_steps,
                                                                           use_updates=True, batch_size=args.batch_size)
        input_parameters = [p.detach() for p in input_parameters]

        if args.simulation == 0:
            rec_machine = inversefed.FedAvgReconstructor(model, (dm, ds), local_steps, args.local_lr, config,
                                                         args.num_images, True, args.batch_size)
        elif args.simulation == 1:
            sim_local_bs = 8
            sim_local_epoch = 2
            sim_local_lr = 0.001
            sim_local_steps = args.num_images // sim_local_bs * sim_local_epoch
            rec_machine = inversefed.FedAvgReconstructor(model, (dm, ds), sim_local_steps, sim_local_lr, config,
                                                         args.num_images, True, sim_local_bs)
        elif args.simulation == 2:
            rec_machine = inversefed.GradientReconstructor(model, (dm, ds), config, num_images=args.num_images)
        else:
            raise ('No support simulation')
        output, stats = rec_machine.reconstruct(input_parameters, labels, img_shape=(3, args.img_shape, args.img_shape))
        ground_truth_all = ground_truth
        output_all = output
        rec_loss = [stats['opt']]

    lpips_scorer = lpips.LPIPS(net="alex").to(**setup)
    # order the reconstruction images
    order = compute_batch_order(lpips_scorer, output_all, ground_truth_all, setup)
    output_all = output_all[order]

    test_mse = (output_all.detach() - ground_truth_all).pow(2).mean()
    test_psnr = inversefed.metrics.psnr(output_all, ground_truth_all, factor=1 / ds)
    test_ssim = inversefed.metrics.cw_ssim(output_all, ground_truth_all, scales=5)
    lpips_score = lpips_scorer(output_all, ground_truth_all, normalize=True)
    avg_lpips = lpips_score.mean().item()

    print(
        f"Rec. loss: {np.mean(rec_loss):2.4f} | MSE: {test_mse:2.4f} | PSNR: {test_psnr:4.2f} | SSIM: {test_ssim:2.4f} | LPIPS: {avg_lpips:2.4f}")
