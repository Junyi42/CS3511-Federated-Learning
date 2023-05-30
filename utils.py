import socket
import dill
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import multiprocessing
from torch.utils.data import Dataset, DataLoader

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
    
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch

def train(model, dataloader, num_epochs, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    for e in range(num_epochs):
        for features, labels in dataloader:
            optimizer.zero_grad()
            probs = model(features)
            loss = criterion(probs, labels)
            loss.backward()
            # update model parameters
            optimizer.step()

def test(global_model, test_loader):
    all_predictions = []
    all_labels = []

    global_model.eval()

    with torch.no_grad():
        for features, labels in test_loader:
            preds = global_model(features)
            predicted_classes = torch.argmax(preds, dim=-1)
            predicted_classes = predicted_classes.tolist()
            labels = labels.tolist()

            all_predictions.extend(predicted_classes)
            all_labels.extend(labels)

    # Convert all prediction results and actual labels to Tensor type
    all_predictions = torch.tensor(all_predictions)
    all_labels = torch.tensor(all_labels)

    accuracy = (all_predictions == all_labels).float().mean().item()
    
    return accuracy
