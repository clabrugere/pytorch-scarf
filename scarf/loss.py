import torch
import torch.nn as nn
import torch.nn.functional as F

# see: https://theaisummer.com/simclr/


class InfoNCE(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, embeddings_corrupted):
        # normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        embeddings_corrupted = F.normalize(embeddings_corrupted, p=2, dim=1)

        # similarity
        logits = embeddings @ embeddings_corrupted.transpose(-2, -1)

        # positive samples
        positive = torch.arange(len(embeddings), device=embeddings.device)

        # compute cross entropy
        loss = F.cross_entropy(logits / self.temperature, positive, reduction="mean")

        return loss
