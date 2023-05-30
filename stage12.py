import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
from utils import MLP
from multiprocessing import Pool, cpu_count
import dill
import torch.nn.functional as F
from loguru import logger
import os
import sys

def get_logger(output_file):
    log_format = "[<green>{time:YYYY-MM-DD HH:mm:ss}</green>] {message}"
    logger.configure(handlers=[{"sink": sys.stderr, "format": log_format}])
    while os.path.exists(output_file):
        output_file = output_file.replace('.log', '_1.log')
    if output_file:
        logger.add(output_file, enqueue=True, format=log_format)
    return logger
logger = get_logger(f'./log_file/result_stage1_2.log')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Set the path of model and dataset
MODEL_PATH = "./models/"
CLIENT_MODEL_PATH = os.path.join(MODEL_PATH, "client_models")
GLOBAL_MODEL_PATH = os.path.join(MODEL_PATH, "server_model.pth")
DATA_PATH = './data'
CLIENT_DATA_PATH = "./private_data"
# mkdirs
if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)
if not os.path.exists(CLIENT_MODEL_PATH):
    os.mkdir(CLIENT_MODEL_PATH)
# Parameters
num_clients = 20
num_rounds = 10
num_epochs = [1, 5, 10, 15, 20, 25]
batch_size = 32
lr = 0.01
num_classes = 10
sweep_m = [5, 10, 15]
mode = ['all', 'partial']

def load_data():
    # Load client datasets
    train_datasets = []
    for i in range(num_clients):
        with open(os.path.join(CLIENT_DATA_PATH, f"Client{i+1}.pkl"), 'rb') as f:
            train_datasets.append(dill.load(f))
    # Load test dataset
    test_dataset = datasets.MNIST(
        DATA_PATH, train=False, download=True, 
        transform=transforms.ToTensor()
    )
    return train_datasets, DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def client_update(client_model, optimizer, train_loader, epoch):
    """
    This function updates/trains the model
    on the client side.
    """
    client_model.train()
    for e in range(epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = client_model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

def average_models(global_model, clients_model):
    """
    This function averages the models of the clients and updates the global model.
    """
    global_dict = global_model.state_dict()
    client_state_dicts = [model.state_dict() for model in clients_model]
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_model[k].float() for client_model in client_state_dicts], 0).mean(dim=0)
    global_model.load_state_dict(global_dict)


def test(global_model, test_loader):
    """
    This function tests the global model on test data and returns test loss and accuracy.
    """
    global_model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = global_model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = correct / len(test_loader.dataset)
    return test_loss, accuracy

def training(num_rounds, num_epochs, mode='all', M=20):
    # Load data
    train_datasets, test_loader = load_data()
    dataloader_client = []
    for i in range(num_clients):
        dataloader_client.append(DataLoader(train_datasets[i], batch_size=batch_size, shuffle=True))
    # Initialize global model parameters
    global_model = MLP().to(device)
    global_model_path = GLOBAL_MODEL_PATH
    torch.save(global_model.state_dict(), global_model_path)
    best_accuracy = 0
    for r in range(num_rounds):
        # Train M clients
        if mode == 'all':
            idx = [i for i in range(len(dataloader_client))]
        else:  # mode == 'partial'
            idx = np.random.choice(len(dataloader_client), M, replace=False)

        for i in tqdm(idx):  # For each client
            local_model = MLP().to(device)
            # Load .pth file
            checkpoint = torch.load(global_model_path)
            local_model.load_state_dict(checkpoint)
            # Load the corresponding dataloader
            dataloader = dataloader_client[i]
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(local_model.parameters(), lr=lr)

            for e in range(num_epochs):
                for features, labels in dataloader:
                    features, labels = features.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = local_model(features)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    # Update model parameters
                    optimizer.step()

            # Save model parameters to .pth file
            save_path = CLIENT_MODEL_PATH+f"/model{i+1}.pth"  # e.g. ./models/client_models/model1.pth ~ model20.pth
            torch.save(local_model.state_dict(), save_path)

        # Calculate average of global model parameters
        avg_model = MLP().to(device)
        avg_model.load_state_dict(torch.load(global_model_path))

        for i in idx:
            client_model_path = CLIENT_MODEL_PATH+f"/model{i+1}.pth"  # e.g. ./models/client_models/model1.pth ~ model20.pth
            client_model = MLP().to(device)
            client_model.load_state_dict(torch.load(client_model_path))
            for avg_param, client_param in zip(avg_model.parameters(), client_model.parameters()):
                avg_param.data += client_param.data

        for avg_param in avg_model.parameters():
            avg_param.data /= len(dataloader_client)

        # Update global model parameters
        global_model.load_state_dict(avg_model.state_dict())

        # Test global model and save it if it is better than the previous one
        test_loss, accuracy = test(global_model, test_loader)

        logger.info(f"Round: {r+1}, Test Loss: {test_loss}, Accuracy: {accuracy}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(global_model.state_dict(), global_model_path)


def main():
    # Stage 1: All clients participate in each round of updates
    logger.info("Stage 1: Training with all clients")
    training(num_rounds, num_epochs[1], mode[0], num_clients)

    # Stage 2: Only m clients participate in each round of updates
    logger.info("Stage 2: Training with m clients")
    for epoch in num_epochs:
        for m in sweep_m:
            logger.info(f"Epoch {epoch}, num_clients = {m}")
            training(num_rounds, epoch, mode[1], m)

if __name__ == "__main__":
    main()
