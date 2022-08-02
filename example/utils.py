import random

import numpy as np
import torch
from tqdm.auto import tqdm


def train_epoch(model, criterion, train_loader, optimizer, device, epoch):
    model.train()
    epoch_loss = 0.0
    batch = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)

    for x_1, x_2 in batch:
        x_1, x_2 = x_1.to(device), x_2.to(device)

        # reset gradients
        optimizer.zero_grad()

        # get embeddings
        emb, emb_corrupted = model(x_1, x_2)

        # compute loss
        loss = criterion(emb, emb_corrupted)
        loss.backward()

        # update model weights
        optimizer.step()

        # log progress
        epoch_loss += x_1.size(0) * loss.item()
        batch.set_postfix({"loss": loss.item()})

    return epoch_loss / len(train_loader.dataset)


def dataset_embeddings(model, loader, device):
    model.eval()
    embeddings = []

    with torch.no_grad():
        for x_1, _ in tqdm(loader):
            x_1 = x_1.to(device)
            embeddings.append(model.get_embeddings(x_1))

    embeddings = torch.cat(embeddings).numpy()

    return embeddings


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
