import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
from dataset import NameDataset
from train import device,NUM_CHARS, HIDDEN_SIZE ,NUM_LAYERS ,NUM_EPOCHS  ,BATCH_SIZE ,droup_out

train_set = NameDataset(r'data\train.tsv', train=True)
val_set = NameDataset(r'data\train.tsv', val=True)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
NUM_CLASS = len(set(train_set.sentiment))
patience=3