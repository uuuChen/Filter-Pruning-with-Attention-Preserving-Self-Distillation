import os

import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def cifar10(batch_size):
    num_classes = 10
    normalize = transforms.Normalize((0.4913, 0.4824, 0.4467), (0.2470, 0.2435, 0.2616))
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)
    return train_loader, val_loader, num_classes


def cifar100(batch_size):
    num_classes = 100
    normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)
    return train_loader, val_loader, num_classes


def cinic10(batch_size):
    num_classes = 10
    data_dir = './data/cinic-10'
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    normalize = transforms.Normalize((0.4789, 0.4723, 0.4305), (0.2421, 0.2383, 0.2587))
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            train_dir,
            transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                normalize,
            ])),
        batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            val_dir,
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
        batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)
    return train_loader, val_loader, num_classes


def imagenet(batch_size):
    num_classes = 1000
    data_dir = './data/ImageNet2012'
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'valid')
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            train_dir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
        ])),
        batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            val_dir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
        batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)
    return train_loader, val_loader, num_classes
