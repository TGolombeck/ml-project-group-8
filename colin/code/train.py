# main file for the project can be used to train the model, test the model, and evaluate the model

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader

from utils.data import ReviewDataset
from models.tokenizer import encode, decode
from models.transformer import Transformer

from models.tokenizer import tokenizer

from sklearn.model_selection import train_test_split


batch_size = 32
epochs = 10
learning_rate = 0.0001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## load data set ##
df = pd.read_csv("data/Rating_Prediction_dataset_preprocessed.csv")
print("Loaded dataset")

## train test split ##
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print("Split dataset into train and test")

## tokenize data ##
train_encodings = encode(train_df["Product_Review"])
test_encodings = encode(test_df["Product_Review"])
print("Tokenized data")

## create datasets ##
train_dataset = ReviewDataset(train_encodings, train_df["Ratings"])
test_dataset = ReviewDataset(test_encodings, test_df["Ratings"])
print("Created datasets")

## create data loaders ##
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print("Created data loaders")

## create model ##
model = Transformer(len(tokenizer.get_vocab()), len(tokenizer.get_vocab()))
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
print("Created model")

## train model ##
print("Training model")
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch in train_loader:
        src = batch["input_ids"].to(device)
        tgt = batch["labels"].to(device)
        output = model(src, tgt, None, None)
        loss = loss_fn(output.view(-1, output.size(-1)), tgt.view(-1))
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch + 1}/{epochs}, loss: {train_loss / len(train_loader):.4f}")

## save model ##
torch.save(model.state_dict(), "models/model.pth")