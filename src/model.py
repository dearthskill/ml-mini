"""Network definition (paper §4.2, Figure 1): 122 -> 500 (sigmoid) -> 24 (softmax)."""
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, n_in=122, n_hidden=500, n_out=24):
        super().__init__()
        self.hidden = nn.Linear(n_in, n_hidden)
        self.out = nn.Linear(n_hidden, n_out)
        for layer in (self.hidden, self.out):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        # Returns logits; softmax is applied inside CrossEntropyLoss / at prediction time.
        return self.out(self.hidden(x).sigmoid())
