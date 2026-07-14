import torch
from torchvision import datasets, transforms


def load_task_data(digit_pair, n_samples=500, train=True):
    transform = transforms.ToTensor()
    mnist = datasets.MNIST(root="./data", train=train, download=True, transform=transform)
    mask = (mnist.targets == digit_pair[0]) | (mnist.targets == digit_pair[1])
    x = mnist.data[mask].float() / 255.0
    y = (mnist.targets[mask] == digit_pair[1]).long()
    return x[:n_samples], y[:n_samples]


# All ten binary digit-pair tasks available for sampling in the validation loop
ALL_DIGIT_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (1, 2), (3, 4), (5, 6), (7, 8), (0, 9)]
