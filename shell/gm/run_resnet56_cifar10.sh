#!/usr/bin/env bash
python3 pgad.py \
    --model resnet56 \
    --dataset cifar10 \
    --lr 0.01 \
    --schedule 60 120 160 \
    --lr-drops 0.2 0.2 0.2 \
    --prune-rates 0.6 \
    --prune-mode filter-gm \
    --prune-interval 1 \
    --s-load-model-path saves/resnet56_cifar10/initial_train/model_epochs_1.pt \
    --use-PFEC \
    --seed 8152 \

