#!/usr/bin/env bash
python3 initial_train.py \
    --dataset cifar10 \
    --model resnet56 \
    --schedule 1 60 120 160 \
    --lr_drops 10 0.2 0.2 0.2 \
    --lr 0.01 \
    --batch_size 128 \
    --seed 8152 \
