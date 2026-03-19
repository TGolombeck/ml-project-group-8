import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import re

class ReviewDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) 
                for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    df = pd.read_csv("data/Rating_Prediction_dataset.csv")[["Product_Review", "Ratings"]]

    df = df.dropna()

    # remove emojis in text
    df["Product_Review"] = df["Product_Review"].apply(lambda x: re.sub(r"[^a-zA-Z0-9\s]", "", x))

    # scale rating from 1-5 to 0-4
    df["Ratings"] = df["Ratings"].apply(lambda x: x - 1)

    df.to_csv("data/Rating_Prediction_dataset_preprocessed.csv", index=False)