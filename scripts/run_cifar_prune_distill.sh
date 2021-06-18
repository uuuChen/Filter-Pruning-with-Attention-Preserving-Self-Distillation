#!/usr/bin/env bash

# MSP + KD
python3 pmsp.py --model resnet56 --dataset cifar10 --prune-rates 0.6 --prune-mode filter-n-g-gm-1 --samp-batches 15 --s-path saves/resnet56_cifar10/initial_train/model_epochs_163.pt --distill msp --seed 8152 --betas 500 --msp-ts 3

# MAT + KD
python3 pmsp.py --model resnet56 --dataset cifar10 --prune-rates 0.6 --prune-mode filter-n-g-gm-1 --samp-batches 15 --s-path saves/resnet56_cifar10/initial_train/model_epochs_163.pt --distill mat --seed 8152 --betas 1000 --mat-ws 3

# LSP + KD
python3 pmsp.py --model resnet56 --dataset cifar10 --prune-rates 0.6 --prune-mode filter-n-g-gm-1 --samp-batches 15 --s-path saves/resnet56_cifar10/initial_train/model_epochs_163.pt --distill lsp --seed 8152 --betas 500 --lsp-ts 4

# LSP2 + KD
python3 pmsp.py --model resnet56 --dataset cifar10 --prune-rates 0.6 --prune-mode filter-n-g-gm-1 --samp-batches 15 --s-path saves/resnet56_cifar10/initial_train/model_epochs_163.pt --distill lsp2 --seed 8152 --betas 30000 --lsp2-ws 4

# AT + KD
#python3 pmsp.py --model resnet56 --dataset cifar10 --prune-rates 0.6 --prune-mode filter-n-g-gm-1 --samp-batches 15 --s-path saves/resnet56_cifar10/initial_train/model_epochs_163.pt --distill at --seed 8152 --betas 1000

# SP + KD
#python3 pmsp.py --model resnet56 --dataset cifar10 --prune-rates 0.6 --prune-mode filter-n-g-gm-1 --samp-batches 15 --s-path saves/resnet56_cifar10/initial_train/model_epochs_163.pt --distill sp --seed 8152 --betas 3000

# KD
#python3 pmsp.py --model resnet56 --dataset cifar10 --prune-rates 0.6 --prune-mode filter-n-g-gm-1 --samp-batches 15 --s-path saves/resnet56_cifar10/initial_train/model_epochs_163.pt --distill kd --seed 8152 --alpha 0.0 --betas 0.9

# GM
#python3 pmsp.py --model resnet56 --dataset cifar10 --prune-rates 0.6 --prune-mode filter-gm --s-path saves/resnet56_cifar10/initial_train/model_epochs_163.pt --seed 8152

# n-g-GM
#python3 pmsp.py --model resnet56 --dataset cifar10 --prune-rates 0.6 --prune-mode filter-n-g-gm-1 --samp-batches 15 --s-path saves/resnet56_cifar10/initial_train/model_epochs_163.pt --seed 8152

# MAT-MSP
#python3 pmsp.py --model resnet56 --dataset cifar10 --prune-rates 0.6 --prune-mode filter-n-g-gm-1 --samp-batches 15 --s-path saves/resnet56_cifar10/initial_train/model_epochs_163.pt --distill msp_mat --seed 8152 --betas 500 1000 --mat-ws 3 --msp-ts 3
