#!/usr/bin/env bash

# MSP
python3 pmsp.py --model resnet56 --dataset cifar10 --prune-rates 0.6 --prune-mode filter-n-g-gm-1 --samp-batches 15 --s-path saves/resnet56_cifar10/initial_train/model_epochs_163.pt --distill msp --seed 8152 --betas 500 --msp-ts 3

# MAT
python3 pmsp.py --model resnet56 --dataset cifar10 --prune-rates 0.6 --prune-mode filter-n-g-gm-1 --samp-batches 15 --s-path saves/resnet56_cifar10/initial_train/model_epochs_163.pt --distill mat --seed 8152 --betas 1000 --mat-ws 3

# LSP
python3 pmsp.py --model resnet56 --dataset cifar10 --prune-rates 0.6 --prune-mode filter-n-g-gm-1 --samp-batches 15 --s-path saves/resnet56_cifar10/initial_train/model_epochs_163.pt --distill lsp --seed 8152 --betas 500 --lsp-ts 4

# AT
#python3 pmsp.py --model resnet56 --dataset cifar10 --prune-rates 0.6 --prune-mode filter-n-g-gm-1 --samp-batches 15 --s-path saves/resnet56_cifar10/initial_train/model_epochs_163.pt --distill at --seed 8152 --betas 1000

# SP
#python3 pmsp.py --model resnet56 --dataset cifar10 --prune-rates 0.6 --prune-mode filter-n-g-gm-1 --samp-batches 15 --s-path saves/resnet56_cifar10/initial_train/model_epochs_163.pt --distill sp --seed 8152 --betas 3000

# GM
#python3 pmsp.py --model resnet56 --dataset cifar10 --prune-rates 0.6 --prune-mode filter-gm --s-path saves/resnet56_cifar10/initial_train/model_epochs_163.pt --seed 8152

# n-g-GM
#python3 pmsp.py --model resnet56 --dataset cifar10 --prune-rates 0.6 --prune-mode filter-n-g-gm-1 --samp-batches 15 --s-path saves/resnet56_cifar10/initial_train/model_epochs_163.pt --seed 8152

# MAT-MSP
#python3 pmsp.py --model resnet56 --dataset cifar10 --prune-rates 0.6 --prune-mode filter-n-g-gm-1 --samp-batches 15 --s-path saves/resnet56_cifar10/initial_train/model_epochs_163.pt --distill msp_mat --seed 8152 --betas 500 1000 --mat-ws 3 --msp-ts 3
