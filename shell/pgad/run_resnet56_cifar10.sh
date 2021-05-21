#!/usr/bin/env bash
python3 pgad.py \
    --dataset cifar10 \
    --model resnet56 \
    --schedule 60 120 160 \
    --lr_drops 0.2 0.2 0.2 \
    --lr 0.01 \
    --prune-rates 0.4 \
    --s-load-model-path saves/resnet56_cifar10/initial_train/model_epochs_1.pt \
    --prune-mode filter-n-g-gm-1 \
    --dist-mode all-grad-dist \
    --use-PFEC
