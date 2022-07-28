import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform


class MLP(torch.nn.Sequential):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout=0.0):

        layers = []
        in_dim = input_dim
        for _ in range(n_layers - 1):
            layers.extend(
                [
                    torch.nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    torch.nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim

        layers.extend([torch.nn.Linear(in_dim, hidden_dim), torch.nn.Dropout(dropout)])

        super().__init__(*layers)


class SCARF(nn.Module):
    def __init__(self, input_dim, hidden_size, low, high, corruption_rate=0.6):
        super().__init__()

        self.encoder = MLP(input_dim, hidden_size, 4)
        self.pretraining_head = MLP(hidden_size, hidden_size, 2)

        # initialize weights
        self.encoder.apply(self._init_weights)
        self.pretraining_head.apply(self._init_weights)

        # uniform distribution over marginal distributions of dataset's features
        self._marginals = Uniform(low, high)
        self._q = int(corruption_rate * input_dim)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)

    def forward(self, x):
        # x: (batch size, m)
        batch_size, m = x.size()

        # create a mask size (batch size, m) where for each sample we set jth column
        # to True randomly so that n_corrupted / m = q
        # create a random tensor of size (batch size, m) drawn from the uniform
        # distribution defined by the min, max values of the marginals of the traning set
        # replace x_c_ij by r_ij where mask_ij is true

        corruption_mask = torch.zeros_like(x, dtype=torch.bool)
        for i in range(batch_size):
            corruption_idx = torch.randperm(m)[: self._q]
            corruption_mask[i, corruption_idx] = True

        x_random = self._marginals.sample((batch_size,))
        x_corrupted = torch.where(corruption_mask, x_random, x)

        # get embeddings
        embeddings = self.encoder(x)
        embeddings = self.pretraining_head(embeddings)

        embeddings_corrupted = self.encoder(x_corrupted)
        embeddings_corrupted = self.pretraining_head(embeddings_corrupted)

        return embeddings, embeddings_corrupted

    def get_embeddings(self, x):
        return self.encoder(x)
