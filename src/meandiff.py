from typing import Dict, Iterator, List, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data import BertWordVectors

FEATURES = [
    "fposx",
    "gaze",
    "firlp",
    "laslp",
    "fixdur",
    "osacdur",
    "osacdx",
    "osacdy",
    "osacl",
    "isacdur",
    "isacdx",
    "isacdy",
]


class MeandiffEncoder(nn.Module):
    def __init__(self, encoding_size: int = 20):
        super().__init__()
        self.encoding_size = encoding_size

        # BERT embeddings -> smaller encodings
        self.encode = nn.Linear(768, encoding_size)
        # Smaller encodings -> eye tracking features
        self.out = nn.Linear(encoding_size, len(FEATURES))

    def forward(self, word_vector) -> torch.Tensor:
        encoding = F.relu(self.encode(word_vector))
        output = self.out(encoding)
        return output


DataIterator = Iterator[Tuple[torch.Tensor, torch.Tensor]]


def iter_data(data: pd.DataFrame, stimuli: Dict[int, List[str]]) -> DataIterator:
    means = (
        data[["sn", "wn", "group"] + FEATURES]
        .groupby(["sn", "wn", "group"])
        .agg(["mean", "std"])
        .reset_index()
    )

    means_per_group = means.pivot(index=["sn", "wn"], columns="group")
    diffs = pd.DataFrame()

    # Standardize features
    features = {column[0] for column in means_per_group.columns}
    for feature in features:
        diffs[feature] = (
            means_per_group[feature, "mean", 1] - means_per_group[feature, "mean", 0]
        ) / means_per_group[feature, "std", 0]
    diffs = diffs.reset_index()

    bert = BertWordVectors()
    word_vectors = {}
    for sn in diffs["sn"].unique():
        for i, word_vector in enumerate(bert.get_word_vectors(stimuli[sn])):
            word_vectors[(sn, i)] = word_vector

    for _, row in diffs.iterrows():
        sn, wn = row["sn"], row["wn"]
        gaze_vector = torch.tensor(row.drop(["sn", "wn"]), dtype=torch.float32)
        word_vector = word_vectors[(sn, wn)]
        yield word_vector, gaze_vector


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, data: DataIterator):
        self.data = list(data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)


def train_encoder(
    data: pd.DataFrame,
    stimuli: Dict[int, List[str]],
    batch_size: int = 16,
    num_epochs: int = 30,
    **kwargs,
) -> MeandiffEncoder:
    print("Training MeandiffEncoder...")
    model = MeandiffEncoder(**kwargs)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    dataset = _Dataset(iter_data(data, stimuli))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    for epoch in range(num_epochs):
        epoch_loss = 0
        for X, y in loader:
            optimizer.zero_grad()
            y_pred = model(X)
            loss = F.mse_loss(y_pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
    #     print(f"Epoch {epoch} done. Loss: {epoch_loss}")
    # print()

    return model
