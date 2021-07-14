#!/usr/bin/env bash

# ASP
python3 pmsp.py --t-model resnet56 --s-model resnet20 --dataset cifar100 --t-path saves/1625594199/model_best.pt --distill asp --seed 8152 --alpha 0.0 --betas 3000 --dev-idx 0 --log-name ASP.txt --lr 0.05 --n-epochs 240 --batch-size 64 --schedule 150 180 210 --lr-drops 0.1 0.1 0.1

# MSP + KD
python3 pmsp.py --t-model resnet56 --dataset cifar10 --prune-rates 0.6 --prune-mode filter-n-g-gm-1 --samp-batches 15 --t-path saves/resnet56_cifar10/initial_train/model_epochs_163.pt --distill msp --seed 8152 --betas 500 --msp-ts 3

# MAT + KD
python3 pmsp.py --t-model resnet56 --dataset cifar10 --prune-rates 0.6 --prune-mode filter-n-g-gm-1 --samp-batches 15 --t-path saves/resnet56_cifar10/initial_train/model_epochs_163.pt --distill mat --seed 8152 --betas 1000 --mat-ws 3

# LSP + KD
python3 pmsp.py --t-model resnet56 --dataset cifar10 --prune-rates 0.6 --prune-mode filter-n-g-gm-1 --samp-batches 15 --t-path saves/resnet56_cifar10/initial_train/model_epochs_163.pt --distill lsp --seed 8152 --betas 300 --lsp-ts 4

# LSP2 + KD
python3 pmsp.py --t-model resnet56 --dataset cifar10 --prune-rates 0.6 --prune-mode filter-n-g-gm-1 --samp-batches 15 --t-path saves/resnet56_cifar10/initial_train/model_epochs_163.pt --distill lsp2 --seed 8152 --betas 300 --lsp2-ws 4

# AT + KD
#python3 pmsp.py --t-model resnet56 --dataset cifar10 --prune-rates 0.6 --prune-mode filter-n-g-gm-1 --samp-batches 15 --t-path saves/resnet56_cifar10/initial_train/model_epochs_163.pt --distill at --seed 8152 --betas 1000

# SP + KD
#python3 pmsp.py --t-model resnet56 --dataset cifar10 --prune-rates 0.6 --prune-mode filter-n-g-gm-1 --samp-batches 15 --t-path saves/resnet56_cifar10/initial_train/model_epochs_163.pt --distill sp --seed 8152 --betas 3000

# KD
#python3 pmsp.py --t-model resnet56 --dataset cifar10 --prune-rates 0.6 --prune-mode filter-n-g-gm-1 --samp-batches 15 --t-path saves/resnet56_cifar10/initial_train/model_epochs_163.pt --distill kd --seed 8152 --alpha 0.0 --betas 0.9

# PFEC
#python3 pmsp.py --t-model resnet56 --dataset cifar10 --prune-rates 0.6 --prune-mode filter-a --t-path saves/resnet56_cifar10/initial_train/model_epochs_163.pt --seed 8152

# SFP
#python3 pmsp.py --t-model resnet56 --dataset cifar10 --prune-rates 0.6 --prune-mode filter-a --soft-prune --prune-interval 1 --t-path saves/resnet56_cifar10/initial_train/model_epochs_163.pt --seed 8152

# FPGM
#python3 pmsp.py --t-model resnet56 --dataset cifar10 --prune-rates 0.6 --prune-mode filter-gm --t-path saves/resnet56_cifar10/initial_train/model_epochs_163.pt --seed 8152

# NGGM
#python3 pmsp.py --t-model resnet56 --dataset cifar10 --prune-rates 0.6 --prune-mode filter-n-g-gm-1 --samp-batches 15 --t-path saves/resnet56_cifar10/initial_train/model_epochs_163.pt --seed 8152

# MAT-MSP
#python3 pmsp.py --t-model resnet56 --dataset cifar10 --prune-rates 0.6 --prune-mode filter-n-g-gm-1 --samp-batches 15 --t-path saves/resnet56_cifar10/initial_train/model_epochs_163.pt --distill msp_mat --seed 8152 --betas 500 1000 --mat-ws 3 --msp-ts 3
