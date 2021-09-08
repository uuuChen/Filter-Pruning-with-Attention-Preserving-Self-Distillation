#!/usr/bin/env bash

# ------------------------
# CIFAR-10
# ------------------------
python3 initial_train.py --model resnet56 --dataset cifar10 --lr 0.01 --schedule 1 60 120 160 --lr-drops 10 0.2 0.2 0.2 --batch-size 128 --seed 8152


# ------------------------
# CIFAR-100
# ------------------------
python3 initial_train.py --model resnet56 --dataset cifar100 --lr 0.01 --schedule 1 60 120 160 --lr-drops 10 0.2 0.2 0.2 --batch-size 128 --seed 8152


# ------------------------
# CINIC-10
# ------------------------
python3 initial_train.py --model resnet56 --dataset cinic10 --n-epochs 140 --lr 0.01 --schedule 100 120 --lr-drops 0.1 0.1 --batch-size 96 --seed 8152

