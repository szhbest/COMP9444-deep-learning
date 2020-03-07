from imdb_dataloader import IMDB
import numpy as np

import torch
import torch.nn as tnn
import torch.optim as topti

from torchtext import data
from torchtext.vocab import GloVe

textField = data.Field(lower=True, include_lengths=True, batch_first=True)
labelField = data.Field(sequential=False)
train, dev = IMDB.splits(textField, labelField, train="train", validation="dev")
print(textField)

textField.build_vocab(train, dev, vectors=GloVe(name="6B", dim=50))
labelField.build_vocab(train, dev)

trainLoader, testLoader = data.BucketIterator.splits((train, dev), shuffle=True, batch_size=64,
                                                     sort_key=lambda x: len(x.text), sort_within_batch=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# j = 0
# for i, batch in enumerate(trainLoader):
#     # Get a batch and potentially send it to GPU memory.
#     inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
#         device), batch.label.type(torch.FloatTensor).to(device)
#     j += 1
#     print(length)
#     print(length.size)
#     if j > 1:
#         break