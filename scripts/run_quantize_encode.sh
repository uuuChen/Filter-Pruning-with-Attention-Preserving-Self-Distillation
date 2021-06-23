#!/usr/bin/env bash
python3 quantize_encode.py --model resnet56 --dataset cifar10 --n-epochs 20 --lr 0.001 --quan-mode conv-quan --load-path saves/resnet56_cifar10/initial_train/model_epochs_163.pt --quan-bits 5
