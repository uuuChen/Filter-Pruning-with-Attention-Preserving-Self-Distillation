#!/usr/bin/env bash
python3 initial_train.py --save_dir saves/alexnet_cifar100 --model alexnet --dataset cifar100 --n_epochs 200 --schedule 50 100 150 --lr_drop 0.1