#!/usr/bin/env bash

# SEED: (8152, 1011, 6162, 2177)

# ------------------------
# CIFAR-10
# ------------------------
# NGGM + MSP + KD
python3 pruning.py --t-model resnet56 --dataset cifar10 --prune-rates 0.6 --prune-mode filter-n-g-gm-1 --samp-batches 15 --t-path saves/resnet56_cifar10/initial_train/model_epochs_163.pt --distill msp --betas 700 --msp-ts 3 --log-name MSP.txt --seed 8152

# NGGM
python3 pruning.py --t-model resnet56 --dataset cifar10 --prune-rates 0.6 --prune-mode filter-n-g-gm-1 --samp-batches 15 --t-path saves/resnet56_cifar10/initial_train/model_epochs_163.pt --log-name NGGM.txt --seed 8152

# AT + KD
python3 pruning.py --t-model resnet56 --dataset cifar10 --prune-rates 0.6 --prune-mode filter-r --t-path saves/resnet56_cifar10/initial_train/model_epochs_163.pt --distill at --betas 1000 --log-name AT.txt --seed 8152

# SP + KD
python3 pruning.py --t-model resnet56 --dataset cifar10 --prune-rates 0.6 --prune-mode filter-r --t-path saves/resnet56_cifar10/initial_train/model_epochs_163.pt --distill sp --betas 3000 --log-name SP.txt --seed 8152

# PFEC
python3 pruning.py --t-model resnet56 --dataset cifar10 --prune-rates 0.6 --prune-mode filter-a --t-path saves/resnet56_cifar10/initial_train/model_epochs_163.pt --log-name PFEC.txt --seed 8152

# SFP
python3 pruning.py --t-model resnet56 --dataset cifar10 --prune-rates 0.6 --prune-mode filter-a --soft-prune --prune-interval 1 --t-path saves/resnet56_cifar10/initial_train/model_epochs_163.pt --log-name SFP.txt --seed 8152

# FPGM
python3 pruning.py --t-model resnet56 --dataset cifar10 --prune-rates 0.6 --prune-mode filter-gm --t-path saves/resnet56_cifar10/initial_train/model_epochs_163.pt --log-name FPGM.txt --seed 8152


# ------------------------
# CIFAR-100
# ------------------------
# NGGM + MSP + KD
python3 pruning.py --t-model resnet56 --dataset cifar100 --prune-rates 0.6 --prune-mode filter-n-g-gm-1 --samp-batches 15 --t-path saves/1625594199/model_best.pt --distill msp --betas 700 --msp-ts 3 --log-name MSP.txt --seed 8152

# NGGM
python3 pruning.py --t-model resnet56 --dataset cifar100 --prune-rates 0.6 --prune-mode filter-n-g-gm-1 --samp-batches 15 --t-path saves/1625594199/model_best.pt --log-name NGGM.txt --seed 8152

# AT + KD
python3 pruning.py --t-model resnet56 --dataset cifar100 --prune-rates 0.6 --prune-mode filter-r --t-path saves/1625594199/model_best.pt --distill at --betas 1000 --log-name AT.txt --seed 8152

# SP + KD
python3 pruning.py --t-model resnet56 --dataset cifar100 --prune-rates 0.6 --prune-mode filter-r --t-path saves/1625594199/model_best.pt --distill sp --betas 3000 --log-name SP.txt --seed 8152

# PFEC
python3 pruning.py --t-model resnet56 --dataset cifar100 --prune-rates 0.6 --prune-mode filter-a --t-path saves/1625594199/model_best.pt --log-name PFEC.txt --seed 8152

# SFP
python3 pruning.py --t-model resnet56 --dataset cifar100 --prune-rates 0.6 --prune-mode filter-a --soft-prune --prune-interval 1 --t-path saves/1625594199/model_best.pt --log-name SFP.txt --seed 8152

# FPGM
python3 pruning.py --t-model resnet56 --dataset cifar100 --prune-rates 0.6 --prune-mode filter-gm --t-path saves/1625594199/model_best.pt --log-name FPGM.txt --seed 8152


# ------------------------
# PLOT FEATURES
# ------------------------
#python3 pruning.py --t-model resnet56  --dataset cifar100 --t-path saves/1625594199/model_best.pt --distill msp --seed 8152 --alpha 0.0 --betas 3000 --dev-idx 0 --msp-ts 3
#python3 pruning.py --t-model resnet56  --dataset cifar100 --t-path saves/1625594199/model_best.pt --distill msp --seed 8152 --alpha 0.0 --betas 3000 --dev-idx 0 --msp-ts 3
#python3 pruning.py --t-model resnet20  --dataset cifar10 --t-path saves/1625413142/model_best.pt --distill msp --seed 8152 --alpha 0.0 --betas 3000 --dev-idx 0 --msp-ts 3
#python3 pruning.py --t-model resnet110  --dataset cifar10 --t-path saves/1625397042/model_best.pt --distill msp --seed 8152 --alpha 0.0 --betas 3000 --dev-idx 0 --msp-ts 3