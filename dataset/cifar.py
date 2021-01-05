import torchvision
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

def rescaling(x): return (x - .5) * 2.

def get_cifar(batch_size=128, num_workers=8, label=None):
    transform = transforms.Compose(
        [transforms.ToTensor(), rescaling])
    trainset = CIFAR10(root='../data', train=True,
                       download=True, transform=transform)
    if label:
        targets_train = torch.tensor(trainset.targets)
        target_train_idx = targets_train == label
        trainset = torch.utils.data.Subset(trainset, np.where(target_train_idx == 1)[0])
    train_loader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=True, num_workers=8)

    testset = CIFAR10(root='../data', train=False,
                      download=True, transform=transform)
    if label:
        targets_test = torch.tensor(testset.targets)
        target_test_idx = targets_test == label
        testset = torch.utils.data.Subset(testset, np.where(target_test_idx == 1)[0])
    test_loader = DataLoader(testset, batch_size=batch_size,
                            shuffle=False, num_workers=8)
    return train_loader, test_loader
