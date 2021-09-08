#!/usr/bin/env bash

# SEED: (8152, 1011, 6162, 2177)

# ------------------------
# CIFAR-10
# ------------------------
# NGGM + MSP + KD
python3 pruning.py --t-model resnet56 --s-copy-t --dataset cifar10 --prune-rates 0.6 --prune-mode filter-n-g-gm-1 --samp-batches 25 --t-path saves/resnet56_cifar10/initial_train/model_epochs_163.pt --distill msp --betas 700 --msp-ts 3 --log-name MSP.txt --seed 8152

# NGGM
python3 pruning.py --t-model resnet56 --s-copy-t --dataset cifar10 --prune-rates 0.6 --prune-mode filter-n-g-gm-1 --samp-batches 25 --t-path saves/resnet56_cifar10/initial_train/model_epochs_163.pt --log-name NGGM.txt --seed 8152

# AT + KD
python3 pruning.py --t-model resnet56 --dataset cifar10 --prune-rates 0.6 --prune-mode filter-r --t-path saves/resnet56_cifar10/initial_train/model_epochs_163.pt --distill at --betas 1000 --log-name AT.txt --seed 8152

# SP + KD
python3 pruning.py --t-model resnet56 --dataset cifar10 --prune-rates 0.6 --prune-mode filter-r --t-path saves/resnet56_cifar10/initial_train/model_epochs_163.pt --distill sp --betas 3000 --log-name SP.txt --seed 8152

# PFEC
python3 pruning.py --t-model resnet56 --s-copy-t --dataset cifar10 --prune-rates 0.6 --prune-mode filter-a --t-path saves/resnet56_cifar10/initial_train/model_epochs_163.pt --log-name PFEC.txt --seed 8152

# SFP
python3 pruning.py --t-model resnet56 --s-copy-t --dataset cifar10 --prune-rates 0.6 --prune-mode filter-a --soft-prune --prune-interval 1 --t-path saves/resnet56_cifar10/initial_train/model_epochs_163.pt --log-name SFP.txt --seed 8152

# FPGM
python3 pruning.py --t-model resnet56 --s-copy-t --dataset cifar10 --prune-rates 0.6 --prune-mode filter-gm --t-path saves/resnet56_cifar10/initial_train/model_epochs_163.pt --log-name FPGM.txt --seed 8152


# ------------------------
# CIFAR-100
# ------------------------
# NGGM + MSP + KD
python3 pruning.py --t-model resnet56 --s-copy-t --dataset cifar100 --prune-rates 0.6 --prune-mode filter-n-g-gm-1 --samp-batches 25 --t-path saves/1625594199/model_best.pt --distill msp --betas 700 --msp-ts 3 --log-name MSP.txt --seed 8152

# NGGM
python3 pruning.py --t-model resnet56 --s-copy-t --dataset cifar100 --prune-rates 0.6 --prune-mode filter-n-g-gm-1 --samp-batches 25 --t-path saves/1625594199/model_best.pt --log-name NGGM.txt --seed 8152

# AT + KD
python3 pruning.py --t-model resnet56 --dataset cifar100 --prune-rates 0.6 --prune-mode filter-r --t-path saves/1625594199/model_best.pt --distill at --betas 1000 --log-name AT.txt --seed 8152

# SP + KD
python3 pruning.py --t-model resnet56 --dataset cifar100 --prune-rates 0.6 --prune-mode filter-r --t-path saves/1625594199/model_best.pt --distill sp --betas 3000 --log-name SP.txt --seed 8152

# PFEC
python3 pruning.py --t-model resnet56 --s-copy-t --dataset cifar100 --prune-rates 0.6 --prune-mode filter-a --t-path saves/1625594199/model_best.pt --log-name PFEC.txt --seed 8152

# SFP
python3 pruning.py --t-model resnet56 --s-copy-t --dataset cifar100 --prune-rates 0.6 --prune-mode filter-a --soft-prune --prune-interval 1 --t-path saves/1625594199/model_best.pt --log-name SFP.txt --seed 8152

# FPGM
python3 pruning.py --t-model resnet56 --s-copy-t --dataset cifar100 --prune-rates 0.6 --prune-mode filter-gm --t-path saves/1625594199/model_best.pt --log-name FPGM.txt --seed 8152


# ------------------------
# CINIC-10
# ------------------------
# NGGM + MSP + KD
python3 pruning.py --t-model resnet56 --s-copy-t --t-path saves/1630871975/model_best.pt --dataset cinic10 --prune-rates 0.6 --prune-mode filter-n-g-gm-1 --samp-batches 25 --distill msp --alpha 0.6 --betas 700 --kd-t 16.0 --msp-ts 3 --n-epochs 140 --lr 0.01 --schedule 100 120 --lr-drops 0.1 0.1 --batch-size 96 --log-name pruned-CINIC10.txt --seed 8152

# NGGM
python3 pruning.py --t-model resnet56 --s-copy-t --t-path saves/1630871975/model_best.pt --dataset cinic10 --prune-rates 0.6 --prune-mode filter-n-g-gm-1 --samp-batches 25 --n-epochs 140 --lr 0.01 --schedule 100 120 --lr-drops 0.1 0.1 --batch-size 96 --log-name pruned-CINIC10.txt --seed 8152

# AT + KD
python3 pruning.py --t-model resnet56 --t-path saves/1630871975/model_best.pt --dataset cinic10 --prune-rates 0.6 --prune-mode filter-r --distill at --alpha 0.6 --betas 50 --kd-t 16.0 --n-epochs 140 --lr 0.01 --schedule 100 120 --lr-drops 0.1 0.1 --batch-size 96 --log-name pruned-CINIC10.txt --seed 8152

# SP + KD
python3 pruning.py --t-model resnet56 --t-path saves/1630871975/model_best.pt --dataset cinic10 --prune-rates 0.6 --prune-mode filter-r --distill sp --alpha 0.6 --betas 2000 --kd-t 16.0 --n-epochs 140 --lr 0.01 --schedule 100 120 --lr-drops 0.1 0.1 --batch-size 96 --log-name pruned-CINIC10.txt --seed 8152

# PFEC
python3 pruning.py --t-model resnet56 --s-copy-t --t-path saves/1630871975/model_best.pt --dataset cinic10 --prune-rates 0.6 --prune-mode filter-a --n-epochs 140 --lr 0.01 --schedule 100 120 --lr-drops 0.1 0.1 --batch-size 96 --log-name pruned-CINIC10.txt --seed 8152

# SFP
python3 pruning.py --t-model resnet56 --s-copy-t --t-path saves/1630871975/model_best.pt --dataset cinic10 --prune-rates 0.6 --prune-mode filter-a --soft-prune --prune-interval 1 --n-epochs 140 --lr 0.01 --schedule 100 120 --lr-drops 0.1 0.1 --batch-size 96 --log-name pruned-CINIC10.txt --seed 8152

# FPGM
python3 pruning.py --t-model resnet56 --s-copy-t --t-path saves/1630871975/model_best.pt --dataset cinic10 --prune-rates 0.6 --prune-mode filter-gm --n-epochs 140 --lr 0.01 --schedule 100 120 --lr-drops 0.1 0.1 --batch-size 96 --log-name pruned-CINIC10.txt --seed 8152


# ------------------------
# ImageNet
# ------------------------
# NGGM + MSP + KD
python3 pruning.py --t-model resnet50 --s-copy-t --dataset imagenet --n-epochs 100 --batch-size 256 --lr 0.1 --schedule 30 60 90 --lr-drops 0.1 0.1 0.1 --prune-rates 0.6 --prune-mode filter-n-g-gm-1 --samp-batches 15 --distill msp --betas 700 --msp-ts 3 --seed 8152

# NGGM
python3 pruning.py --t-model resnet50 --s-copy-t --dataset imagenet --n-epochs 100 --batch-size 256 --lr 0.1 --schedule 30 60 90 --lr-drops 0.1 0.1 0.1 --prune-rates 0.6 --prune-mode filter-n-g-gm-1 --samp-batches 15 --seed 8152

# AT + KD
python3 pruning.py --t-model resnet50 --dataset imagenet --n-epochs 100 --batch-size 256 --lr 0.1 --schedule 30 60 90 --lr-drops 0.1 0.1 0.1 --prune-rates 0.6 --prune-mode filter-r --distill at --betas 1000 --seed 8152

# SP + KD
python3 pruning.py --t-model resnet50 --dataset imagenet --n-epochs 100 --batch-size 256 --lr 0.1 --schedule 30 60 90 --lr-drops 0.1 0.1 0.1 --prune-rates 0.6 --prune-mode filter-r --distill sp --betas 3000 --seed 8152

# PFEC
python3 pruning.py --t-model resnet50 --s-copy-t --dataset imagenet --n-epochs 100 --batch-size 256 --lr 0.1 --schedule 30 60 90 --lr-drops 0.1 0.1 0.1 --prune-rates 0.6 --prune-mode filter-a  --seed 8152

# SFP
python3 pruning.py --t-model resnet50 --s-copy-t --dataset imagenet --n-epochs 100 --batch-size 256 --lr 0.1 --schedule 30 60 90 --lr-drops 0.1 0.1 0.1 --prune-rates 0.6 --prune-mode filter-a --soft-prune --prune-interval 1 --seed 8152

# FPGM
python3 pruning.py --t-model resnet50 --s-copy-t --dataset imagenet --n-epochs 100 --batch-size 256 --lr 0.1 --schedule 30 60 90 --lr-drops 0.1 0.1 0.1 --prune-rates 0.6 --prune-mode filter-gm --seed 8152


# ------------------------
# PLOT FEATURES
# ------------------------
#python3 pruning.py --t-model resnet56  --dataset cifar100 --t-path saves/1625594199/model_best.pt --distill msp --seed 8152 --alpha 0.0 --betas 3000 --dev-idx 0 --msp-ts 3
#python3 pruning.py --t-model resnet56  --dataset cifar100 --t-path saves/1625594199/model_best.pt --distill msp --seed 8152 --alpha 0.0 --betas 3000 --dev-idx 0 --msp-ts 3
#python3 pruning.py --t-model resnet20  --dataset cifar10 --t-path saves/1625413142/model_best.pt --distill msp --seed 8152 --alpha 0.0 --betas 3000 --dev-idx 0 --msp-ts 3
#python3 pruning.py --t-model resnet110  --dataset cifar10 --t-path saves/1625397042/model_best.pt --distill msp --seed 8152 --alpha 0.0 --betas 3000 --dev-idx 0 --msp-ts 3