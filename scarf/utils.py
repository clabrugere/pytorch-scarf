import random

import numpy as np
import torch
from tqdm.auto import tqdm


def train_epoch(model, criterion, device, train_loader, optimizer, epoch):
    model.train()
    epoch_loss = 0.0
    batch = tqdm(train_loader, desc=f"Epoch {epoch}")

    for input, target in batch:
        input, _ = input.to(device), target.to(device)

        # reset gradients
        optimizer.zero_grad()

        # get embeddings
        emb, emb_corrupted = model(input)

        # compute loss
        loss = criterion(emb, emb_corrupted)
        loss.backward()

        # update model weights
        optimizer.step()

        # log progress
        epoch_loss += input.size(0) * loss.item()
        batch.set_postfix({"loss": loss.item()})

    batch.set_postfix({"loss": epoch_loss / len(train_loader.dataset)})

    return epoch_loss / len(train_loader.dataset)


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
