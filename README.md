# Filter Pruning with Attention-Preserving Self-Distillation


## Overview
NGGM (Filter Pruning)          |  HAP (Self-Distillation)
:-------------------------:|:-------------------------:
<img src="https://i.imgur.com/PoE65ur.png" alt="drawing" width="500"/>  |  <img src="https://i.imgur.com/qhMekAx.png" alt="drawing" width="700"/>

* Benchmarks 3 state-of-the-art filter pruning methods in PyTorch:

    Paper (Filter Pruning)| Name
    :---|:-----|
    FPEC (ICLR'17)  | Pruning Filters for Efficient ConvNets 
    SFP (IJCAI'18)  | Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks
    FPGM (CVPR'19) | Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration
    
* Benchmarks 3 state-of-the-art knowledge distillation methods in PyTorch:

    Paper (Distillation)| Name
    :---|:-----|
    AT (ICLR'17)  | Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer  
    SP (ICCV'19)  | Similarity-Preserving Knowledge Distillation  
    AFD (AAAA'21) | Show, Attend and Distill: Knowledge Distillation via Attention-based Feature Matching
    

## Requirements
* Python (>3.6)
* PyTorch (>1.7.1)
* torchVision 
* numpy
* tensorboardx
* sklearn 
* tqdm

## Running
### Vanilla ResNet Training 
* Running commands in `scripts/run_vanilla.sh`.  An example of running `ResNet-56` on `CIFAR-10` is given by:
    ```bash
    python3 initial_train.py --model resnet56 --dataset cifar10 --lr 0.01 --schedule 1 60 120 160 --lr-drops 10 0.2 0.2 0.2 --batch-size 128 --seed 8152
    ```
    where the flags are explained as:
    * `--schedule`: specify at which epoch to drop the learning rate.
    * `--lr-drops`: specify how much the learning rate should be multiplied by the epoch corresponding to the schedule. 
    * _Note: the length of `--schedule` and `--lr-drops` should be same_.
     
### Pruned ResNet Training 
* Running commands in `scripts/run_pruning.sh`.  An example of running `ResNet-56` on `CIFAR-10` and using distillation method `AT (ICLR'17)` is given by:
    ```bash
    python3 pruning.py --t-model resnet56 --s-copy-t --dataset cifar10 --prune-rates 0.6 --prune-mode filter-r --t-path saves/1625594011/model_best.pt --distill at --betas 1000 --log-name PRUNED-CIFAR10.txt --seed 8152
    ```
    where the flags are explained as:
    * `--t-model`: specify the model used by the teacher.
    * `--s-copy-t`:  copy the parameters of the pre-trained teacher model as the student initialization parameters. _Note: it can not be used for distillation of different architectures._
    * `--prune-rates`: specify the proportion of filters to be retained for each convolutional layer, default: `1.0`, i.e. no filters will be pruned by default.
    * `--prune-mode`: specify what pruning method to use, including:
        * `filter-r`: prune randomly.
        * `filter-a`: prune by L1-norm of the filter, i.e. `PFEC (ICLR'17)`.
        * `filter-gm`: prune by geometric-median, i.e. `FGPM (CVPR'19)`.
        * `filter-nggm`: prune by our method.
    * `--t-path`: pre-trained model corresponding to `--t-model`.
    * `--distill`: specify what distillation method to use, including:
        * `at`: `AT (ICLR'17)`.
        * `sp`: `SP (ICCV'19)`.
        * `afd`: `AFD (AAAI'21)`.
        * `hap`: our method.
        * _Note: by default, we add `KD (NIPS'14)` to all the baselines_.
    * `--log-name`: specify the name of the log file. By default, the log file will be saved at `./saves` directory. 
 
### Quantized ResNet Training + Huffman Coding
 * Running commands in `scripts/run_quantization_encode.sh`. 
 * _Note: we ensure the accuracies of the model before huffman encoding and after decoding are the same to ensure the correctness of our implementation._.
### Benchmark Results on CIFAR-100
<img src="https://i.imgur.com/7ziVCD8.png" alt="drawing"/>

* _Note: The value in `Acc. after pruning (%)` column is the mean of four experiments with different but static seeds._
