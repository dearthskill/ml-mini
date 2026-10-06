"""Training loop (paper §4.2): mini-batch SGD, lr 0.5, batch 1000, L2 weight decay."""
import numpy as np
import torch
import torch.nn as nn

from model import Net

LR = 0.5
BATCH_SIZE = 1000
WEIGHT_DECAY = 1e-4


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(X_train, y_train, epochs, seed, device, on_epoch_end=None):
    set_seed(seed)
    model = Net().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.0,
                                weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()
    X = torch.from_numpy(X_train).to(device)
    y = torch.from_numpy(y_train).to(device)
    gen = torch.Generator().manual_seed(seed)
    n = len(X)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=gen).to(device)
        total = 0.0
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            optimizer.zero_grad()
            loss = loss_fn(model(X[idx]), y[idx])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        if on_epoch_end is not None:
            on_epoch_end(epoch, model, total / n)
    return model
