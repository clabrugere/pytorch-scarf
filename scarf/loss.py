import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCE(nn.Module):
    def __init__(self, temperature=1.0, reduction="mean"):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, embeddings, embeddings_corrupted):
        # normalize the embedding vector so that they have unit norm
        embeddings = F.normalize(embeddings, p=2, dim=1)
        embeddings_corrupted = F.normalize(embeddings_corrupted, p=2, dim=1)

        # compute pairwise cosine similarity betwteen embeddings
        similarity = embeddings @ embeddings_corrupted.transpose(-2, -1)

        # positive samples
        positive = torch.arange(len(embeddings), device=embeddings.device)

        # compute mean cross entropy
        loss = F.cross_entropy(
            similarity / self.temperature, positive, reduction=self.reduction
        )

        return loss
