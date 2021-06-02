#!/usr/bin/env bash
python3 pgad.py \
    --model resnet56 \
    --dataset cifar10 \
    --batch-size 128 \
    --lr 0.01 \
    --schedule 60 120 160 \
    --lr-drops 0.2 0.2 0.2 \
    --prune-rates 0.6 \
    --prune-mode filter-n-g-gm-1 \
    --samp-batches 10 \
    --dist-mode all-dist \
    --s-load-model-path saves/resnet56_cifar10/initial_train/model_epochs_163.pt \
    --seed 8152 \

