# Pytorch-Scarf

Implementation of [SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption](https://arxiv.org/abs/2106.15147) in Pytorch.

The model learns a representation of tabular data using contrastive learning. It is inspired from SimCLR and uses a similar architecture and loss.

# Install

Clone the repo

```git clone https://github.com/clabrugere/pytorch-scarf.git```

or install from the repo

```pip install git+https://github.com/clabrugere/pytorch-scarf.git```

# Usage

``` python
from scarf.loss import NTXent
from scarf.model import SCARF


# preprocess your data and create your pytorch dataset
# train_ds = ...

# train the model
batch_size = 128
epochs = 5000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

model = SCARF(
    input_dim=train_ds.shape[1],
    emb_dim=16,
    corruption_rate=0.5,
).to(device)
optimizer = Adam(model.parameters(), lr=0.001)
ntxent_loss = NTXent()

for epoch in range(1, epochs + 1):
  for anchor, positive in train_loader:
        anchor, positive = anchor.to(device), positive.to(device)

        # reset gradients
        optimizer.zero_grad()

        # get embeddings
        emb, emb_corrupted = model(anchor, positive)

        # compute loss
        loss = ntxent_loss(emb, emb_corrupted)
        loss.backward()

        # update model weights
        optimizer.step()
```

For more details, refer to the example notebook `example/example.ipynb` and how to supply samples to the model in `example/dataset.py`

# Parameters

Model:

- `input_dim` (int): dimension of a sample
- `emb_dim` (int): dimension of the embedding vector
- `encoder_depth` (int): number of layers in the encoder MLP. Defaults to 4
- `head_depth` (int): number of layers in the pre-training head. Defaults to 2
- `corruption_rate` (float): fraction of features to corrupt. Defaults to 0.6
- `encoder` (nn.Module): encoder network. Defaults to None
- `pretraining_head`(nn.Module): pre-training head network. Defaults to None


NT-Xent loss:

- `temperature` (float):

# SCARF

![Architecture](resources/architecture.png)

This model builds embeddings of tabular data in a self-supervised fashion similarly to SimCLR using a contrastive approach.

For each sample (anchor) in a batch of size N, a positive view is synthetically built by corrupting from the anchor a fixed amount of features drawn randomly each time, hence giving a final batch of size 2N.

The corruption is made by simply replacing feature's values by the one observed in another sample that is drawn randomly. There is no explicitly defined negative views, but instead the procedure considers the 2(N - 1) other examples in the batch as negative views.

An network `f`, the encoder, learns a representation of the anchor and positive view that is then fed to a projection head `g`that is not used to generate the embeddings.

The learning procedure is about maximizing similarity between the anchor and the positive sample using NT-Xent loss from SimCLR. The similarity used is the cosine similarity.

# Citations

```
@misc{https://doi.org/10.48550/arxiv.2106.15147,
  doi = {10.48550/ARXIV.2106.15147},

  url = {https://arxiv.org/abs/2106.15147},

  author = {Bahri, Dara and Jiang, Heinrich and Tay, Yi and Metzler, Donald},

  keywords = {Machine Learning (cs.LG), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},

  title = {SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption},

  publisher = {arXiv},

  year = {2021},

  copyright = {Creative Commons Attribution 4.0 International}
}
```

```
@misc{https://doi.org/10.48550/arxiv.2002.05709,
  doi = {10.48550/ARXIV.2002.05709},

  url = {https://arxiv.org/abs/2002.05709},

  author = {Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey},

  keywords = {Machine Learning (cs.LG), Computer Vision and Pattern Recognition (cs.CV), Machine Learning (stat.ML), FOS: Computer and information sciences, FOS: Computer and information sciences},

  title = {A Simple Framework for Contrastive Learning of Visual Representations},

  publisher = {arXiv},

  year = {2020},

  copyright = {arXiv.org perpetual, non-exclusive license}
}

```
