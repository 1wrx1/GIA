# batch size 1
python main_learn_dlg.py --lr 1e-4 --epochs 200 --leak_mode None --model MLP-3000 --dataset CIFAR10 --batch_size 256 --shared_model LeNet --data_root $data_root --checkpoint_root $checkpoint_root

# batch size > 1
python main_learn_dlg.py --lr 1e-4 --epochs 5000 --leak_mode batch-4 --model MLP-10000 --dataset CIFAR10 --batch_size 256 --shared_model LeNet --data_root $data_root --checkpoint_root $checkpoint_root
python main_learn_dlg.py --lr 1e-4 --epochs 5000 --leak_mode batch-8 --model MLP-10000 --dataset CIFAR10 --batch_size 256 --shared_model LeNet --data_root $data_root --checkpoint_root $checkpoint_root
python main_learn_dlg.py --lr 1e-4 --epochs 5000 --leak_mode batch-16 --model MLP-10000 --dataset CIFAR10 --batch_size 256 --shared_model LeNet --data_root $data_root --checkpoint_root $checkpoint_root
python main_learn_dlg.py --lr 1e-4 --epochs 5000 --leak_mode batch-32 --model MLP-10000 --dataset CIFAR10 --batch_size 256 --shared_model LeNet --data_root $data_root --checkpoint_root $checkpoint_root