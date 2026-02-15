import torch
import torch.nn as nn
import pandas as pd
import pickle

from preprocess import preprocess_text

df = pd.read_csv("data/emails.csv")

df["clean_text"] = df["text"].apply(preprocess_text)

vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

X = vectorizer.transform(df["clean_text"]).toarray()

labels = df["label"].astype("category").cat.codes
y = labels.values

X = torch.tensor(X).float()
y = torch.tensor(y).long()

class EmailNN(nn.Module):

    def __init__(self, input_size, num_classes):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

input_size = X.shape[1]
num_classes = len(set(y.tolist()))

model = EmailNN(input_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(20):

    outputs = model(X)

    loss = criterion(outputs, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    print("Epoch:", epoch, "Loss:", loss.item())

torch.save(model.state_dict(), "models/dl_model.pth")
