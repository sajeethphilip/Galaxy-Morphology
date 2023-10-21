'''
This code integrates W&B, logs important metrics, saves the best model using W&B, and provides enhanced visibility into your training process and results. Remember to replace "your-project-name" with your actual W&B project name, and configure W&B with your API key and email as needed for authentication.
'''
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import os
import csv
from PIL import Image
import wandb  # Import W&B

# Hyperparameters
batch_size = 16
num_epochs = 100
learning_rate = 0.001

# Data paths
Operation = input("Train  or Predict? (Default : Predict ): ")
if Operation == 'Train':
    Train = True
    Predict = False
    data_root = input("Enter the location of training data  (Default: Data): ")
    data_test_root = input("Enter the location of test data  (Default: Data_test): ")
    if data_root == '':
        data_root = "Data"
    if data_test_root == '':
        data_test_root = "Data_test"
else:
    Train = False
    Predict = True
    data_root = input("Enter the location of Prediction  data  (Default: Predict): ")
    data_test_root = input("Enter the location of test data  (Default: Data_test): ")
    if data_root == '':
        data_root = "Predict"
    if data_test_root == '':
        data_test_root = "Data_test"

best_model_path = "Best_TransferLearning_model.pth"
labels_path = "model_labels.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
map_location = torch.device(device)

# Initialize W&B
wandb.init(project="your-project-name",  # Replace with your W&B project name
           config={
               "batch_size": batch_size,
               "learning_rate": learning_rate,
               # Add other hyperparameters here
           })

# Transformations for data augmentation and normalization
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data_root, transform=None):
        self.data_root = data_root
        self.transform = transform
        self.class_names = sorted(os.listdir(data_root))
        self.image_names = []
        self.labels = []

        for class_idx, class_name in enumerate(self.class_names):
            class_folder = os.path.join(data_root, class_name)
            self.image_names.extend([os.path.join(class_name, img_name) for img_name in os.listdir(class_folder)])
            self.labels.extend([class_idx] * len(os.listdir(class_folder)))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        img_path = os.path.join(self.data_root, self.image_names[index])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[index]
        return image, label, os.path.basename(self.image_names[index])

# Load pre-trained model
model = models.resnet18(pretrained=True)

# Training loop
train_dataset = CustomDataset(data_root=data_root, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

if Train:
    # Datasets and data loaders
    num_classes = len(train_dataset.class_names)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_accuracy = 0
    if os.path.exists(best_model_path):
        model = torch.load(best_model_path, map_location=torch.device(device))
        print("Loaded pre-trained model.")
    model.to(device)
    model.train()
    # Save labels associated with the model during training
    if not os.path.exists(best_model_path):
        model.labels = os.listdir(data_root)
        torch.save(model.labels, labels_path)
    else:
        model.labels = torch.load(labels_path, map_location=torch.device(device))
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels, image_names in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if (1 - running_loss / len(train_loader)) > best_accuracy:
            best_accuracy = (1 - running_loss / len(train_loader))
            # Save the best model using W&B
            if epoch > 0:
                torch.save(model, wandb.run.dir + "/best_model.pth")
                print("Best model saved.")
        # Log loss and other metrics to W&B
        wandb.log({
            "epoch": epoch,
            "loss": running_loss / len(train_loader),
            "best_accuracy": best_accuracy,
        })
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

model = torch.load(best_model_path, map_location=torch.device(device))

# Testing loop
model.eval()
test_dataset = CustomDataset(data_root=data_test_root, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
model_labels = np.array(torch.load(labels_path, map_location=torch.device(device)))

if Predict:
    csv_file = "predictions.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name\t\t\t", "prediction\t", "probability"])

    with torch.no_grad(), open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        for images, labels, image_names in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            for img_name, label_idx, pred_idx, prob in zip(image_names, labels, predicted, probabilities):
                label_name = train_dataset.class_names[label_idx]
                pred_name = model_labels[pred_idx.item()]
                prob_value = prob[pred_idx].item()

                row = [data_test_root + "/" + label_name + "/" + img_name + "\t", pred_name + "\t", prob_value]
                writer.writerow(row)

# Validation accuracy
correct = 0
total = 0
with torch.no_grad():
    for images, labels, image_names in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        t_labels = os.listdir(data_test_root)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        for pred_idx, label_idx in zip(predicted, labels):
            label_name = test_dataset.class_names[label_idx]
            if model_labels[pred_idx.item()] == label_name:
                correct += 1

accuracy = 100 * correct / total
print(f"Validation Accuracy: {accuracy:.2f}%")

# Finish the W&B run
wandb.finish()
