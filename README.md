# Exploring the Vulnerabilities of Federated Learning:
A Deep Dive into Gradient Inversion Attacks

## Usage

### 1. Preparation

#### 1.1 Clone the repo.

```bash
git clone https://github.com/1wrx1/GIA.git
cd GIA
```

#### 1.2 Install the environment

```bash
conda env create -f environment.yml 
conda activate GIA 
```

### 2. Evaluation
#### 2.1 OP-GIA
##### 2.1.1 IG
For example, evaluating the attack performance of IG on the ImageNet dataset with a batch size of 64 using an untrained model.

```bash
cd OP-GIA/IG
python -u inverting.py --dataset ImageNet --num_classes 1000 --img_shape 224 --batch_size 64 --gpu 0 --root $root
```
by setting `$root` as root of ImageNet.

More commands are provided in `run.sh`.

#### 2.2 GEN-GIA
##### 2.2.1 GGL

```bash
cd GEN-GIA/GGL
```
A notebook example of running GGL on the ImageNet dataset is provided in `ImageNet-ng-My-All-Images.ipynb`.

A notebook example of running GGL on the ImageNet dataset under practical scenario is provided in `ImageNet-ng-My-All-Images-FedAvg.ipynb`.

A notebook example of running GGL on the CIFAR-100 dataset is provided in `CIFAR-100-ng-My-All-Images.ipynb`.

##### 2.2.2 CI-Net

For example, evaluating the attack performance of CI-Net on the ImageNet dataset with a batch size of 64 on ResNet18 with sigmoid activation layers.

```bash
cd GEN-GIA/CI-Net
python -u main.py --dataset imagenet --root $root --arch resnet18 --bs 64 --act sigmoid
```
by setting `$root` as root of ImageNet.

More commands are provided in `run.sh`.

##### 2.2.3 LTI

For example, evaluating the attack performance of LTI on the CIFAR10 dataset with a batch size of 1 on LeNet.

```bash
cd GEN-GIA/LTI
python -u main_learn_dlg.py --lr 1e-4 --epochs 200 --leak_mode None --model MLP-3000 --dataset CIFAR10 --batch_size 256 --shared_model LeNet --data_root $data_root --checkpoint_root $checkpoint_root
```
by setting `$data_root` as root of CIFAR10 and `$checkpoint_root` as the root to save the checkpoints.

A notebook to evaluate the results is provided in `Results--Vision.ipynb`

More commands are provided in `run.sh`.

#### 2.3 ANA-GIA
##### 2.3.1 Robbing the Fed

```bash
cd ANA-GIA/RoF
```
A notebook example is provided in `breaching_fl-My-All-Images.ipynb`.

##### 2.3.2 Fishing

For example, evaluating the attack performance of Fishing on the ImageNet dataset with a batch size of 64 on ResNet18.

```bash
cd ANA-GIA/fish
python -u fish.py --bs 64 --resolution 224 --arch resnet18 --root $root
```
by setting `$root` as root of ImageNet.

More commands are provided in `run.sh`.

## Acknowledgement
We many thank the authors for releasing their codes: [IG](https://github.com/JonasGeiping/invertinggradients), [GGL](https://github.com/zhuohangli/GGL), [CI-Net](https://github.com/czhang024/CI-Net), [LTI](https://github.com/wrh14/Learning_to_Invert), [Robbing the Fed](https://github.com/lhfowl/robbing_the_fed), and [Fishing](https://github.com/JonasGeiping/breaching).


