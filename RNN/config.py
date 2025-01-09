import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
from dataset import NameDataset

device = torch.device('cuda:0')
NUM_CHARS = 128
HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_EPOCHS = 15
BATCH_SIZE = 512
droup_out =0.5
train_set = NameDataset(r'E:\senmtiment-analysis\train.tsv\train.tsv', train=True)
val_set = NameDataset(r'E:\senmtiment-analysis\train.tsv\train.tsv', val=True)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
NUM_CLASS = len(set(train_set.sentiment))
patience=3