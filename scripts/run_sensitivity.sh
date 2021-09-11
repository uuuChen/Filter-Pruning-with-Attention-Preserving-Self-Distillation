#!/usr/bin/env bash

seeds=(8152 1011 6162 2177)
betas=(100 300 500 700 900 1100)
gammas=(0.0 0.2 0.4 0.6 0.8)


# Beta
for seed in "${seeds[@]}"
do
    for beta in "${betas[@]}"
    do
        python3 pruning.py --t-model resnet56 --s-copy-t --dataset cifar100 --prune-rates 0.6 --prune-mode filter-n-g-gm-1 --samp-batches 25 --t-path saves/1625594199/model_best.pt --distill msp --log-name SENSITIVITY.txt --seed "$seed" --betas "$beta"
    done
done


# Gamma
for seed in "${seeds[@]}"
do
    for gamma in "${gammas[@]}"
    do
        python3  pruning.py --t-model resnet56 --s-copy-t --dataset cifar100 --prune-rates 0.6 --prune-mode filter-n-g-gm-1 --samp-batches 25 --t-path saves/1625594199/model_best.pt --distill msp --betas 700 --log-name SENSITIVITY.txt --seed "$seed" --gamma "$gamma"
    done
done

