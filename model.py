import os
import itertools
import csv
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import random
from collections import Counter
import numpy as np
import time
import gc

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device_type = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 18
torch.manual_seed(18)
random.seed(18)
np.random.seed(18)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(18)
    torch.manual_seed(18)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# give training set more distortion to discourage overfitting
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1), #slightly distort. food classification probably requires color signals so can't distort to much. 
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# show mostly normal test images to model to see if it generalized well
test_transform = transforms.Compose([
    transforms.Resize(144),
    transforms.RandomCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load datasets with correct transforms
train_dataset = datasets.ImageFolder(root="train", transform=train_transform) 
test_dataset = datasets.ImageFolder(root="test", transform=test_transform)

# Read an example image
#image, label = train_dataset[84]
#print(image.shape, label)

# Set num_workers 
num_workers = min(14, os.cpu_count())  # auto-detect cpu core count, cap at 14 (lower this cap if system crashes)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True, worker_init_fn=seed_worker, generator=g)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True, worker_init_fn=seed_worker, generator=g)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FoodCNN(nn.Module):
    def __init__(self, plot_graph=True, early_stopping=True, patience=5, eval_interval=5):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4 
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.AdaptiveAvgPool2d((1, 1))  # Output: (batch_size, 512, 1, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 91)
        )

        # control flags
        self.plot_graph = plot_graph
        self.early_stopping = early_stopping # saves time is model stops improving
        self.patience = patience # threshold to stop training if accuracy doesnt improve enough accross epochs
        self.eval_interval = eval_interval # gather test and train accuracy every n epochs for plotting
        
    def forward(self, X):
        X = self.features(X)
        X = self.classifier(X)
        return X

    def _train_and_save_model(self):
        pass  
    
    def plot_accuracies(self, train_accuracies, test_accuracies):
        """Plots training and testing accuracies after training, and saves plot."""
        if not os.path.exists('plots'):
            os.makedirs('plots')
    
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"accuracy_plot_{timestamp}.png"
        save_path = os.path.join('plots', filename)
    
        epochs = len(train_accuracies)
        x_values = np.linspace(1, 100, epochs)
    
        plt.figure(figsize=(8, 6))
        plt.plot(x_values, train_accuracies, marker='o', label='Train Accuracy')
        plt.plot(x_values, test_accuracies, marker='x', label='Test Accuracy')
        plt.title("Accuracy vs Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.xticks(np.arange(0, 101, 10))  # Ticks at 0, 10, ..., 100
        plt.legend()
        plt.grid(True)
    
        plt.savefig(save_path)
    
        if self.plot_graph:
            plt.show()

def calculate_test_accuracy(model, test_loader): 
    model.eval()  # set model to evaluation mode
    test_correct = 0
    total_test = 0

    with torch.no_grad():  # disable gradient computation (faster)
        for X_test, y_test in test_loader:
            X_test = X_test.to(device)
            y_test = y_test.to(device)
            
            #with autocast(dtype=torch.float16):
            with autocast(enabled=torch.cuda.is_available()):
                outputs = model(X_test)
            
            predictions = torch.argmax(outputs, dim=1)
            test_correct += (predictions == y_test).sum().item()
            total_test += y_test.size(0)

    test_accuracy = (test_correct / total_test) * 100
    return test_accuracy

def _train_and_save_model(self):
    self.to(device)
    torch.manual_seed(18)
    epochs = 100  # or however many you want

    #ADAM
    #optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) # learning rate gradaully decreases naturally (fight overfitting)
    #SGD
    optimizer = torch.optim.SGD(self.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # prevent overconfidence with label smoothing
    scaler = GradScaler()

    best_test_accuracy = 0.0
    train_accuracies = []
    test_accuracies = []
    patience_counter = 0  # for early stopping

    print("Device being used:", device)
    start_time = time.time()

    for e in range(epochs):
        self.train()
        training_correct = 0
        total_train = 0

        for X_train, y_train in tqdm(train_loader, desc=f"Epoch {e+1}"):
            X_train, y_train = X_train.to(device), y_train.to(device)
            optimizer.zero_grad()

            #with autocast(dtype=torch.float16):
            with autocast(enabled=torch.cuda.is_available()):
                y_pred = self(X_train)
                loss = criterion(y_pred, y_train)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            label_pred = torch.argmax(y_pred, dim=1)
            training_correct += (label_pred == y_train).sum().item()
            total_train += y_train.size(0)
            
        scheduler.step()

        # Evaluate test and train accuracy every 'self.eval_interval' epochs
        if (e + 1) % self.eval_interval == 0 or e == epochs - 1:
            train_accuracy = (training_correct / total_train) * 100
            print(f"Training accuracy of epoch {e+1}: {train_accuracy:.2f}%")
            train_accuracies.append(train_accuracy)
            
            test_accuracy = calculate_test_accuracy(self, test_loader)
            test_accuracies.append(test_accuracy)
            print(f"Test accuracy of epoch {e+1}: {test_accuracy:.2f}%")

            # Save best model
            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
                torch.save(self.state_dict(), "best_model.pth")
                patience_counter = 0  # reset patience counter
            else:
                patience_counter += 1
            
            # Early stopping check
            if self.early_stopping and patience_counter >= self.patience:
                print(f"Early stopping triggered at epoch {e+1}.")
                break
        
        torch.cuda.empty_cache()

    end_time = time.time()
    print(f"Training Time: {end_time - start_time:.2f} seconds")

    # Plot at the end
    self.plot_accuracies(train_accuracies, test_accuracies)

FoodCNN._train_and_save_model = _train_and_save_model

#import gc
#import torch

gc.collect()               # Python garbage collector: clears unused CPU memory
torch.cuda.empty_cache() 

# Instantiate and train a new model
model = FoodCNN().to(device)
model._train_and_save_model()  
        
