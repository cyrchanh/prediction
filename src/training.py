import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def train(model, x, y, epochs=5, lr=1e-3):
    loader = DataLoader(TensorDataset(x, y), batch_size=64, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
    return model


def accuracy(model, x, y):
    with torch.no_grad():
        preds = model(x).argmax(dim=1)
        return (preds == y).float().mean().item()
