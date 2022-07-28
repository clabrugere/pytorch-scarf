import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform


class MLP(torch.nn.Sequential):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout=0.0):

        layers = []
        in_dim = input_dim
        for _ in range(n_layers - 1):
            layers.append(torch.nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(torch.nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.extend([torch.nn.Linear(in_dim, hidden_dim), torch.nn.Dropout(dropout)])

        super().__init__(*layers)


class SCARF(nn.Module):
    def __init__(
        self,
        input_dim,
        emb_dim,
        marginals_min,
        marginals_max,
        encoder_depth=4,
        head_depth=2,
        corruption_rate=0.6,
    ):
        super().__init__()

        self.encoder = MLP(input_dim, emb_dim, encoder_depth)
        self.pretraining_head = MLP(emb_dim, emb_dim, head_depth)

        # initialize weights
        self.encoder.apply(self._init_weights)
        self.pretraining_head.apply(self._init_weights)

        # uniform distribution over marginal distributions of dataset's features
        self._marginals = Uniform(marginals_min, marginals_max)
        self.corruption_len = int(corruption_rate * input_dim)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)

    def forward(self, x):
        # x: (batch size, m)
        batch_size, m = x.size()

        # 1: create a mask of size (batch size, m) where for each sample we set the jth column
        # to True at random, such that corruption_len / m = corruption_rate
        # 2: create a random tensor of size (batch size, m) drawn from the uniform
        # distribution defined by the min, max values of the marginals of the traning set
        # 3: replace x_corrupted_ij by x_random_ij where mask_ij is true

        corruption_mask = torch.zeros_like(x, dtype=torch.bool)
        for i in range(batch_size):
            corruption_idx = torch.randperm(m)[: self.corruption_len]
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
