#cifar10
python main.py --dataset cifar10 --root $root --arch resnet18 --bs 64 --act $act
python main.py --dataset cifar10 --root $root --arch resnet18 --bs 32 --act $act
python main.py --dataset cifar10 --root $root --arch resnet18 --bs 16 --act $act
python main.py --dataset cifar10 --root $root --arch resnet18 --bs 8 --act $act
python main.py --dataset cifar10 --root $root --arch resnet18 --bs 4 --act $act
python main.py --dataset cifar10 --root $root --arch resnet18 --bs 1 --act $act

#cifar100
python main.py --dataset cifar100 --root $root --arch resnet18 --bs 64 --act $act
python main.py --dataset cifar100 --root $root --arch resnet18 --bs 32 --act $act
python main.py --dataset cifar100 --root $root --arch resnet18 --bs 16 --act $act
python main.py --dataset cifar100 --root $root --arch resnet18 --bs 8 --act $act
python main.py --dataset cifar100 --root $root --arch resnet18 --bs 4 --act $act
python main.py --dataset cifar100 --root $root --arch resnet18 --bs 1 --act $act


#ImageNet(resolution 128) change imagenet to imagenet64/imagenet256 for different resolutions
python main.py --dataset imagenet --root $root --arch resnet18 --bs 64 --act $act
python main.py --dataset imagenet --root $root --arch resnet18 --bs 32 --act $act
python main.py --dataset imagenet --root $root --arch resnet18 --bs 16 --act $act
python main.py --dataset imagenet --root $root --arch resnet18 --bs 8 --act $act
python main.py --dataset imagenet --root $root --arch resnet18 --bs 4 --act $act
python main.py --dataset imagenet --root $root --arch resnet18 --bs 1 --act $act

#celeba64
python main.py --dataset celeba64 --root $root --arch resnet18 --bs 64 --act $act
python main.py --dataset celeba64 --root $root --arch resnet18 --bs 32 --act $act
python main.py --dataset celeba64 --root $root --arch resnet18 --bs 16 --act $act
python main.py --dataset celeba64 --root $root --arch resnet18 --bs 8 --act $act
python main.py --dataset celeba64 --root $root --arch resnet18 --bs 4 --act $act
python main.py --dataset celeba64 --root $root --arch resnet18 --bs 1 --act $act

#PEFT $model : vit-large-LoRA/vit-base-LoRA/vit-tiny-LoRA
python main.py --dataset $dataset --root $root--arch $model --bs $bs