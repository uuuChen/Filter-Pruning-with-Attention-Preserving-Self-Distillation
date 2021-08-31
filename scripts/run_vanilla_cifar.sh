#!/usr/bin/env bash

# Resnet56 - CIFAR-10
python3 initial_train.py --model resnet56 --dataset cifar10 --lr 0.01 --schedule 1 60 120 160 --lr-drops 10 0.2 0.2 0.2 --batch-size 128 --seed 8152

# Resnet56 - CIFAR-100
python3 initial_train.py --model resnet56 --dataset cifar100 --lr 0.01 --schedule 1 60 120 160 --lr-drops 10 0.2 0.2 0.2 --batch-size 128 --seed 8152



