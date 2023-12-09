import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform


class MLP(torch.nn.Sequential):
    def __init__(self, input_dim, hidden_dim, num_hidden, dropout=0.0):
        layers = []
        in_dim = input_dim
        for _ in range(num_hidden - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.extend([nn.Linear(in_dim, hidden_dim), nn.Dropout(dropout)])

        super().__init__(*layers)


class SCARF(nn.Module):
    def __init__(
        self,
        input_dim,
        emb_dim,
        features_low,
        features_high,
        num_hidden=4,
        head_depth=2,
        corruption_rate=0.6,
        dropout=0.0,
    ):
        super().__init__()

        self.encoder = MLP(input_dim, emb_dim, num_hidden, dropout)
        self.pretraining_head = MLP(emb_dim, emb_dim, head_depth)

        # uniform distribution over marginal distributions of dataset's features
        self.marginals = Uniform(torch.Tensor(features_low), torch.Tensor(features_high))
        self.corruption_len = int(corruption_rate * input_dim)

    def forward(self, x):
        batch_size, m = x.size()

        # 1: create a mask of size (batch size, m) where for each sample we set the jth column to True at random, such that corruption_len / m = corruption_rate
        # 2: create a random tensor of size (batch size, m) drawn from the uniform distribution defined by the min, max values of the training set
        # 3: replace x_corrupted_ij by x_random_ij where mask_ij is true

        corruption_mask = torch.zeros_like(x, dtype=torch.bool, device=x.device)
        for i in range(batch_size):
            corruption_idx = torch.randperm(m)[: self.corruption_len]
            corruption_mask[i, corruption_idx] = True

        x_random = self.marginals.sample(torch.Size((batch_size,))).to(x.device)
        x_corrupted = torch.where(corruption_mask, x_random, x)

        # get embeddings
        embeddings = self.encoder(x)
        embeddings = self.pretraining_head(embeddings)

        embeddings_corrupted = self.encoder(x_corrupted)
        embeddings_corrupted = self.pretraining_head(embeddings_corrupted)

        return embeddings, embeddings_corrupted

    @torch.inference_mode()
    def get_embeddings(self, x):
        return self.encoder(x)
