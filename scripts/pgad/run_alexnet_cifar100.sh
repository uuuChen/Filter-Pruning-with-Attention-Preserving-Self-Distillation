#!/usr/bin/env bash
python3 pgad.py \
    --model alexnet \
    --dataset cifar100 \
    --s-load-model-path saves/alexnet_cifar100/initial_train/model_epochs_0.pt \
    --prune-rates 0.84 0.38 0.35 0.37 0.37 \
    --prune-mode filter-gm \
    --dist-mode all-grad-dist \
    --use-PFEC \


