import os
from PIL import Image
import numpy as np
from random import random
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import timm
from ..BenchmarkArk.dataloader import ChestXray14, build_transform_classification
# Define the device to be used (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define your dataset and data loaders
# Example: Assuming you have CIFAR-10 dataset, you can replace this with your dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to fit SWIN input size
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images
])

dataset_train = ChestXray14(
            images_path="/anvil/scratch/x-ssiingh/JLiangLab/datasets/nih_xray14/nih_xray14/images/images",
            file_path="/anvil/scratch/x-ssiingh/JLiangLab//BenchmarkArk/dataset/Xray14_train_official.txt",
            augment=build_transform_classification(
                normalize="imagenet",
                mode="train",
                crop_size=224,
                resize=224,
            ),
            annotation_percent=100,
        )
dataset_val = ChestXray14(
    images_path="/anvil/scratch/x-ssiingh/JLiangLab/datasets/nih_xray14/nih_xray14/images/images",
    file_path="/anvil/scratch/x-ssiingh/JLiangLab//BenchmarkArk/dataset/Xray14_val_official.txt",
    augment=build_transform_classification(
        normalize="imagenet",
        mode="valid",
        crop_size=224,
        resize=224,
    ),
)
dataset_test = ChestXray14(
            images_path="/anvil/scratch/x-ssiingh/JLiangLab/datasets/nih_xray14/nih_xray14/images/images",
            file_path="/anvil/scratch/x-ssiingh/JLiangLab//BenchmarkArk/dataset/Xray14_test_official.txt",
            augment=build_transform_classification(
                normalize="imagenet",
                mode="test",
                crop_size=224,
                resize=224,
            ),
        )

train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=32, shuffle=False)

# Define SWIN model
model = timm.create_model('swin_base_patch4_window7_224', pretrained=False)

checkpoint = torch.load("/anvil/scratch/x-ssiingh/JLiangLab/ACE/models/ACE_contrast_12n_global_inequal_swinb.pth", map_location='cpu')
state_dict = checkpoint['student']
state_dict = {k.replace("module.backbone.", ""): v for k, v in state_dict.items()}

model.load_state_dict(state_dict)

# Modify the final layer for your classification task
num_classes = 10  # Example: CIFAR-10 has 10 classes
model.head = nn.Linear(model.head.in_features, num_classes)

# Move model to device
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total

    print(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Train Loss: {train_loss:.4f}, '
          f'Train Accuracy: {train_accuracy:.2f}%')

# Evaluate the model
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

test_accuracy = 100. * correct / total
print(f'Test Accuracy: {test_accuracy:.2f}%')
