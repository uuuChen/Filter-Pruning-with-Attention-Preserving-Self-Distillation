#!/usr/bin/env bash
python3 initial_train.py \
    --dataset cifar10 \
    --model resnet56 \
    --schedule 60 120 160 \
    --lr_drops 0.2 0.2 0.2 \
    --lr 0.1 \
