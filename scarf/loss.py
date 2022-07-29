import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCE(nn.Module):
    def __init__(self, temperature=1.0, reduction="mean"):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, embeddings, embeddings_corrupted):
        # compute pairwise cosine similarity betwteen embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        embeddings_corrupted = F.normalize(embeddings_corrupted, p=2, dim=1)

        similarity = torch.mm(embeddings, embeddings.transpose(0, 1))

        # positive samples
        positive = torch.arange(len(embeddings), device=embeddings.device)

        # compute mean cross entropy
        loss = F.cross_entropy(
            similarity / self.temperature, positive, reduction=self.reduction
        )

        return loss
