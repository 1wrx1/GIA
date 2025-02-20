# resnet18, cifar10, untrained
nohup python -u inverting.py --batch_size 1 --gpu 3 > out/resnet18_cifar10_bs_1.out &
nohup python -u inverting.py --batch_size 4 --gpu 5 > out/resnet18_cifar10_bs_4.out &
nohup python -u inverting.py --batch_size 8 --gpu 1 > out/resnet18_cifar10_bs_8.out &
nohup python -u inverting.py --batch_size 16 --gpu 6 > out/resnet18_cifar10_bs_16.out &
nohup python -u inverting.py --batch_size 32 --gpu 2 > out/resnet18_cifar10_bs_32.out &
nohup python -u inverting.py --batch_size 64 --gpu 4 > out/resnet18_cifar10_bs_64.out &

# resnet18, cifar100, untrained
nohup python -u inverting.py --dataset CIFAR100 --num_classes 100 --batch_size 1 --gpu 3 > out/resnet18_cifar100_bs_1.out &
nohup python -u inverting.py --dataset CIFAR100 --num_classes 100 --batch_size 4 --gpu 2 > out/resnet18_cifar100_bs_4.out &
nohup python -u inverting.py --dataset CIFAR100 --num_classes 100 --batch_size 8 --gpu 5 > out/resnet18_cifar100_bs_8.out &
nohup python -u inverting.py --dataset CIFAR100 --num_classes 100 --batch_size 16 --gpu 4 > out/resnet18_cifar100_bs_16.out &
nohup python -u inverting.py --dataset CIFAR100 --num_classes 100 --batch_size 32 --gpu 1 > out/resnet18_cifar100_bs_32.out &
nohup python -u inverting.py --dataset CIFAR100 --num_classes 100 --batch_size 64 --gpu 6 > out/resnet18_cifar100_bs_64.out &

# resnet18, imagenet, untrained
nohup python -u inverting.py --dataset ImageNet --num_classes 1000 --img_shape 224 --batch_size 1 --gpu 6 > out/resnet18_imagenet_bs_1_repeat3.out &
nohup python -u inverting.py --dataset ImageNet --num_classes 1000 --img_shape 224 --batch_size 4 --gpu 5 > out/resnet18_imagenet_bs_4_repeat3.out &
nohup python -u inverting.py --dataset ImageNet --num_classes 1000 --img_shape 224 --batch_size 8 --gpu 0 > out/resnet18_imagenet_bs_8_repeat3.out &
nohup python -u inverting.py --dataset ImageNet --num_classes 1000 --img_shape 224 --batch_size 16 --gpu 4 > out/resnet18_imagenet_bs_16_repeat3.out &
nohup python -u inverting.py --dataset ImageNet --num_classes 1000 --img_shape 224 --batch_size 32 --gpu 2 > out/resnet18_imagenet_bs_32_repeat3.out &
nohup python -u inverting.py --dataset ImageNet --num_classes 1000 --img_shape 224 --batch_size 64 --gpu 3 > out/resnet18_imagenet_bs_64_repeat3.out &

# resnet18, cifar10, trained
nohup python -u inverting.py --batch_size 1 --gpu 7 --trained True > out/resnet18_cifar10_bs_1_trained_try3.out &
nohup python -u inverting.py --batch_size 4 --gpu 2 --trained True > out/resnet18_cifar10_bs_4_trained_try3.out &
nohup python -u inverting.py --batch_size 8 --gpu 3 --trained True > out/resnet18_cifar10_bs_8_trained_try3.out &
nohup python -u inverting.py --batch_size 16 --gpu 4 --trained True > out/resnet18_cifar10_bs_16_trained_try3.out &
nohup python -u inverting.py --batch_size 32 --gpu 5 --trained True > out/resnet18_cifar10_bs_32_trained_try3.out &
nohup python -u inverting.py --batch_size 64 --gpu 6 --trained True > out/resnet18_cifar10_bs_64_trained_try3.out &

# resnet18, cifar100, trained
nohup python -u inverting.py --dataset CIFAR100 --num_classes 100 --batch_size 1 --gpu 6 --trained True --file resnet18_CIFAR100_epoch_120.pth > out/resnet18_cifar100_bs_1_trained_try3.out &
nohup python -u inverting.py --dataset CIFAR100 --num_classes 100 --batch_size 4 --gpu 5 --trained True --file resnet18_CIFAR100_epoch_120.pth > out/resnet18_cifar100_bs_4_trained_try3.out &
nohup python -u inverting.py --dataset CIFAR100 --num_classes 100 --batch_size 8 --gpu 4 --trained True --file resnet18_CIFAR100_epoch_120.pth > out/resnet18_cifar100_bs_8_trained_try3.out &
nohup python -u inverting.py --dataset CIFAR100 --num_classes 100 --batch_size 16 --gpu 3 --trained True --file resnet18_CIFAR100_epoch_120.pth > out/resnet18_cifar100_bs_16_trained_try3.out &
nohup python -u inverting.py --dataset CIFAR100 --num_classes 100 --batch_size 32 --gpu 2 --trained True --file resnet18_CIFAR100_epoch_120.pth > out/resnet18_cifar100_bs_32_trained_try3.out &
nohup python -u inverting.py --dataset CIFAR100 --num_classes 100 --batch_size 64 --gpu 1 --trained True --file resnet18_CIFAR100_epoch_120.pth > out/resnet18_cifar100_bs_64_trained_try3.out &

# resnet18, imagenet, trained
nohup python -u inverting.py --dataset ImageNet --num_classes 1000 --img_shape 224 --batch_size 1 --gpu 4 --trained True --file resnet18_ImageNet_epoch_120.pth > out/resnet18_imagenet_bs_1_trained_deflr_0.00001.out &
nohup python -u inverting.py --dataset ImageNet --num_classes 1000 --img_shape 224 --batch_size 4 --gpu 7 --trained True --file resnet18_ImageNet_epoch_120.pth > out/resnet18_imagenet_bs_4_trained_deflr_0.00001.out &
nohup python -u inverting.py --dataset ImageNet --num_classes 1000 --img_shape 224 --batch_size 8 --gpu 5 --trained True --file resnet18_ImageNet_epoch_120.pth > out/resnet18_imagenet_bs_8_trained_deflr_0.00001.out &
nohup python -u inverting.py --dataset ImageNet --num_classes 1000 --img_shape 224 --batch_size 16 --gpu 1 --trained True --file resnet18_ImageNet_epoch_120.pth > out/resnet18_imagenet_bs_16_trained_deflr_0.00001.out &
nohup python -u inverting.py --dataset ImageNet --num_classes 1000 --img_shape 224 --batch_size 32 --gpu 2 --trained True --file resnet18_ImageNet_epoch_120.pth > out/resnet18_imagenet_bs_32_trained_deflr_0.00001.out &
nohup python -u inverting.py --dataset ImageNet --num_classes 1000 --img_shape 224 --batch_size 64 --gpu 3 --trained True --file resnet18_ImageNet_epoch_120.pth > out/resnet18_imagenet_bs_64_trained_deflr_0.00001.out &

# resnet34/50/101, cifar100, untrained
nohup python -u inverting.py --arch ResNet152 --dataset CIFAR100 --num_classes 100 --batch_size 1 --gpu 7 > out/resnet152_cifar100_bs_1.out &
nohup python -u inverting.py --arch ResNet152 --dataset CIFAR100 --num_classes 100 --batch_size 4 --gpu 2 > out/resnet152_cifar100_bs_4.out &
nohup python -u inverting.py --arch ResNet152 --dataset CIFAR100 --num_classes 100 --batch_size 8 --gpu 2 > out/resnet152_cifar100_bs_8.out &
nohup python -u inverting.py --arch ResNet152 --dataset CIFAR100 --num_classes 100 --batch_size 16 --gpu 3 > out/resnet152_cifar100_bs_16.out &
nohup python -u inverting.py --arch ResNet152 --dataset CIFAR100 --num_classes 100 --batch_size 32 --gpu 3 > out/resnet152_cifar100_bs_32.out &
nohup python -u inverting.py --arch ResNet152 --dataset CIFAR100 --num_classes 100 --batch_size 64 --gpu 5 > out/resnet152_cifar100_bs_64.out &

# attacking FedAvg, strong(simulation=0), weak(simulation=1), no simulation (simulation=2)
# epochs: 1/2/5, batch_size: 1/8/16
nohup python -u inverting.py --attack_fedavg True --simulation 2 --epochs 1 --batch_size 1 --gpu 1 > out/resnet18_cifar10_epo_1_bs_1_sim_2.out &
nohup python -u inverting.py --attack_fedavg True --simulation 2 --epochs 1 --batch_size 8 --gpu 1 > out/resnet18_cifar10_epo_1_bs_8_sim_2.out &
nohup python -u inverting.py --attack_fedavg True --simulation 2 --epochs 1 --batch_size 16 --gpu 4 > out/resnet18_cifar10_epo_1_bs_16_sim_2.out &
nohup python -u inverting.py --attack_fedavg True --simulation 2 --epochs 2 --batch_size 1 --gpu 3 > out/resnet18_cifar10_epo_2_bs_1_sim_2.out &
nohup python -u inverting.py --attack_fedavg True --simulation 2 --epochs 2 --batch_size 8 --gpu 3 > out/resnet18_cifar10_epo_2_bs_8_sim_2.out &
nohup python -u inverting.py --attack_fedavg True --simulation 2 --epochs 2 --batch_size 16 --gpu 4 > out/resnet18_cifar10_epo_2_bs_16_sim_2.out &
nohup python -u inverting.py --attack_fedavg True --simulation 2 --epochs 5 --batch_size 1 --gpu 6 > out/resnet18_cifar10_epo_5_bs_1_sim_2.out &
nohup python -u inverting.py --attack_fedavg True --simulation 2 --epochs 5 --batch_size 8 --gpu 5 > out/resnet18_cifar10_epo_5_bs_8_sim_2.out &
nohup python -u inverting.py --attack_fedavg True --simulation 2 --epochs 5 --batch_size 16 --gpu 4 > out/resnet18_cifar10_epo_5_bs_16_sim_2.out &
# CIFAR-100
nohup python -u inverting.py --dataset CIFAR100 --num_classes 100 --attack_fedavg True --simulation 0 --epochs 1 --batch_size 1 --gpu 3 > out/resnet18_cifar100_epo_1_bs_1_sim_0.out &
nohup python -u inverting.py --dataset CIFAR100 --num_classes 100 --attack_fedavg True --simulation 0 --epochs 1 --batch_size 8 --gpu 4 > out/resnet18_cifar100_epo_1_bs_8_sim_0.out &
nohup python -u inverting.py --dataset CIFAR100 --num_classes 100 --attack_fedavg True --simulation 0 --epochs 1 --batch_size 16 --gpu 3 > out/resnet18_cifar100_epo_1_bs_16_sim_0.out &
nohup python -u inverting.py --dataset CIFAR100 --num_classes 100 --attack_fedavg True --simulation 0 --epochs 2 --batch_size 1 --gpu 4 > out/resnet18_cifar100_epo_2_bs_1_sim_0.out &
nohup python -u inverting.py --dataset CIFAR100 --num_classes 100 --attack_fedavg True --simulation 0 --epochs 2 --batch_size 8 --gpu 3 > out/resnet18_cifar100_epo_2_bs_8_sim_0.out &
nohup python -u inverting.py --dataset CIFAR100 --num_classes 100 --attack_fedavg True --simulation 0 --epochs 2 --batch_size 16 --gpu 4 > out/resnet18_cifar100_epo_2_bs_16_sim_0.out &
nohup python -u inverting.py --dataset CIFAR100 --num_classes 100 --attack_fedavg True --simulation 0 --epochs 5 --batch_size 1 --gpu 6 > out/resnet18_cifar100_epo_5_bs_1_sim_0.out &
nohup python -u inverting.py --dataset CIFAR100 --num_classes 100 --attack_fedavg True --simulation 0 --epochs 5 --batch_size 8 --gpu 4 > out/resnet18_cifar100_epo_5_bs_8_sim_0.out &
nohup python -u inverting.py --dataset CIFAR100 --num_classes 100 --attack_fedavg True --simulation 0 --epochs 5 --batch_size 16 --gpu 3 > out/resnet18_cifar100_epo_5_bs_16_sim_0.out &

# defense
# cifar-10
nohup python -u inverting.py --defense noise --d_param 1e-1 --batch_size 1 --gpu 3 > out/resnet18_cifar10_bs_1_noise_1e-1.out &
nohup python -u inverting.py --defense noise --d_param 1e-1 --batch_size 8 --gpu 1 > out/resnet18_cifar10_bs_8_noise_1e-1.out &
nohup python -u inverting.py --defense noise --d_param 1e-1 --batch_size 32 --gpu 2 > out/resnet18_cifar10_bs_32_noise_1e-1.out &
# cifar-100
nohup python -u inverting.py --dataset CIFAR100 --num_classes 100 --defense noise --d_param 1e-1 --batch_size 1 --gpu 3 > out/resnet18_cifar100_bs_1_noise_1e-1.out &
nohup python -u inverting.py --dataset CIFAR100 --num_classes 100 --defense noise --d_param 1e-1 --batch_size 8 --gpu 2 > out/resnet18_cifar100_bs_8_noise_1e-1.out &
nohup python -u inverting.py --dataset CIFAR100 --num_classes 100 --defense noise --d_param 1e-1 --batch_size 32 --gpu 1 > out/resnet18_cifar100_bs_32_noise_1e-1.out &
# imagenet
nohup python -u inverting.py --dataset ImageNet --num_classes 1000 --img_shape 224 --defense noise --d_param 1e-4 --batch_size 1 --gpu 2 > out/resnet18_imagenet_bs_1_noise_1e-4.out &
nohup python -u inverting.py --dataset ImageNet --num_classes 1000 --img_shape 224 --defense noise --d_param 1e-4 --batch_size 8 --gpu 3 > out/resnet18_imagenet_bs_8_noise_1e-4.out &
nohup python -u inverting.py --dataset ImageNet --num_classes 1000 --img_shape 224 --defense noise --d_param 1e-4 --batch_size 32 --gpu 1 > out/resnet18_imagenet_bs_32_noise_1e-4.out &