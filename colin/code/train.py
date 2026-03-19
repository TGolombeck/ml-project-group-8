# main file for the project can be used to train the model, test the model, and evaluate the model

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader

from utils.data import ReviewDataset
from models.tokenizer import encode, decode
from models.transformer import Transformer

batch_size = 32
epochs = 10
learning_rate = 0.0001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## load data set ##
df = pd.read_csv("data/Rating_Prediction_dataset.csv")[["Product_Review", "Ratings"]]

## train test split ##
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

## tokenize data ##
train_encodings = encode(train_df["Product_Review"])
test_encodings = encode(test_df["Product_Review"])

## create datasets ##
train_dataset = ReviewDataset(train_encodings, train_df["Ratings"])
test_dataset = ReviewDataset(test_encodings, test_df["Ratings"])

## create data loaders ##
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

## create model ##
model = Transformer(len(tokenizer.get_vocab()), len(tokenizer.get_vocab()))

## train model ##
for epoch in range(epochs):
    train_loss = 0
    for batch in train_loader:
        src = batch["input_ids"].to(device)
        tgt = batch["labels"].to(device)
        src_mask = batch["attention_mask"].to(device)
        tgt_mask = batch["attention_mask"].to(device)

        output = model(src, tgt, src_mask, tgt_mask)
        loss = loss_fn(output, tgt)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

