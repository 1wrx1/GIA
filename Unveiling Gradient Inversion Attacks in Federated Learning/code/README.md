# Unveiling Gradient Inversion Attacks in Federated Learning

## Requirements

Some important required packages are lised below:

- Python 3.10
- PyTorch 2.0.1
- torchvision 0.15.2
- lpips 0.1.4
- nevergrad 1.0.2


## Usage

### 1. Create a conda environment

```bash
cd ./FL_privacy
conda create -n fl_privacy python=3.10
conda activate fl_privacy
pip install -r requirements.txt
```

### 2. Evaluation
#### IG
For example, evaluating the attack performance of IG on the ImageNet dataset with a batch size of 64 using an untrained model.
```bash
cd ./IG
python -u inverting.py --dataset ImageNet --num_classes 1000 --img_shape 224 --batch_size 64 --gpu 0
```
More commands are provided in `run.sh`.


#### GGL

```bash
cd ./GGL
```
A notebook example of running GGL on the ImageNet dataset is provided in `ImageNet-ng-My-All-Images.ipynb`.

A notebook example of running GGL on the ImageNet dataset under practical scenario is provided in `ImageNet-ng-My-All-Images-FedAvg.ipynb`.

A notebook example of running GGL on the CIFAR-100 dataset is provided in `CIFAR-100-ng-My-All-Images.ipynb`.


#### Robbing the Fed

```bash
cd ./RoF
```
A notebook example is provided in `breaching_fl-My-All-Images.ipynb`.


## Acknowledgement
We many thank the authors for releasing their codes: [IG](https://github.com/JonasGeiping/invertinggradients), [GGL](https://github.com/zhuohangli/GGL), and [Robbing the Fed](https://github.com/lhfowl/robbing_the_fed).
