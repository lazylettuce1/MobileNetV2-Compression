"""
data.py — CIFAR-10 loading, normalization, and augmentation.

Train transforms:
    RandomCrop(32, padding=4, reflect)  -> standard CIFAR translation augmentation
    RandomHorizontalFlip()              -> CIFAR classes are flip-invariant
    ToTensor() + Normalize(mean, std)   -> per-channel CIFAR-10 statistics
    RandomErasing()                     -> cheap cutout-style regularizer, no extra epochs needed

Test transforms:
    ToTensor() + Normalize(mean, std) only — deterministic evaluation, no crop/flip.
"""
import torch
import torchvision
import torchvision.transforms as T

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_transforms():
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4, padding_mode="reflect"),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        T.RandomErasing(p=0.25, scale=(0.02, 0.2)),
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return train_tf, test_tf


def get_dataloaders(data_dir: str = "./data",
                     batch_size: int = 128,
                     num_workers: int = 4):
    train_tf, test_tf = get_transforms()

    train_set = torchvision.datasets.CIFAR10(
        data_dir, train=True, download=True, transform=train_tf
    )
    test_set = torchvision.datasets.CIFAR10(
        data_dir, train=False, download=True, transform=test_tf
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader
