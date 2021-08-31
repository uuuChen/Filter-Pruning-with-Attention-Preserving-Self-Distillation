#!/usr/bin/env bash

# NGGM + MSP + KD
python3 pruning.py --n-epochs 100 --betch-size 256 --lr 0.1 --schedule 30 60 90 --lr-drops 0.1 0.1 0.1 --t-model resnet50 --dataset imagenet --prune-rates 0.6 --prune-mode filter-n-g-gm-1 --samp-batches 15 --distill msp --betas 700 --msp-ts 3 --seed 8152

# NGGM
python3 pruning.py --n-epochs 100 --betch-size 256 --lr 0.1 --schedule 30 60 90 --lr-drops 0.1 0.1 0.1 --t-model resnet50 --dataset imagenet --prune-rates 0.6 --prune-mode filter-n-g-gm-1 --samp-batches 15 --seed 8152

# AT + KD
python3 pruning.py --n-epochs 100 --betch-size 256 --lr 0.1 --schedule 30 60 90 --lr-drops 0.1 0.1 0.1 --t-model resnet50 --dataset imagenet --prune-rates 0.6 --prune-mode filter-r --distill at --betas 1000 --seed 8152

# SP + KD
python3 pruning.py --n-epochs 100 --betch-size 256 --lr 0.1 --schedule 30 60 90 --lr-drops 0.1 0.1 0.1 --t-model resnet50 --dataset imagenet --prune-rates 0.6 --prune-mode filter-r --distill sp --betas 3000 --seed 8152

# PFEC
python3 pruning.py --n-epochs 100 --betch-size 256 --lr 0.1 --schedule 30 60 90 --lr-drops 0.1 0.1 0.1 --t-model resnet50 --dataset imagenet --prune-rates 0.6 --prune-mode filter-a  --seed 8152

# SFP
python3 pruning.py --n-epochs 100 --betch-size 256 --lr 0.1 --schedule 30 60 90 --lr-drops 0.1 0.1 0.1 --t-model resnet50 --dataset imagenet --prune-rates 0.6 --prune-mode filter-a --soft-prune --prune-interval 1 --seed 8152

# FPGM
python3 pruning.py --n-epochs 100 --betch-size 256 --lr 0.1 --schedule 30 60 90 --lr-drops 0.1 0.1 0.1  --t-model resnet50 --dataset imagenet --prune-rates 0.6 --prune-mode filter-gm --seed 8152



