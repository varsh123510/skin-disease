import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Path to your dataset
data_dir = r"C:\Users\TOSHIBA\Downloads\archive\Split_smol\val"

# Define transforms (resize, convert to tensor, normalize)
transform = transforms.Compose([
    transforms.Resize((128, 128)),   # resize all images
    transforms.ToTensor(),           # convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # normalize pixels
])

# Load dataset
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Create DataLoader (for batching)
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Check dataset
print("Classes:", dataset.classes)   # list of disease names
print("Number of images:", len(dataset))

# Fetch one batch
images, labels = next(iter(data_loader))
print("Batch image shape:", images.shape)
print("Batch label shape:", labels.shape)

# defining a model
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"
num_classes = len(dataset.classes)  # number of skin disease categories

# ✅ Updated model loading (no warnings)
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

# Replace the classifier to match your number of classes
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
#prepare data loaders
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Path to the parent dataset folder
data_dir = r"C:\Users\TOSHIBA\Downloads\archive\Split_smol"

# Transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Datasets
train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
val_dataset = datasets.ImageFolder(root=f"{data_dir}/val", transform=transform)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
# training the model
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
data_dir = r"C:\Users\TOSHIBA\Downloads\archive\Split_smol"

# Transforms with data augmentation for training
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Datasets
train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=train_transform)
val_dataset = datasets.ImageFolder(root=f"{data_dir}/val", transform=val_transform)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Model
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
num_classes = len(train_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 10
best_val_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_train_loss = running_loss / len(train_loader)
    
    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f} - Val Acc: {val_acc:.2f}%")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_skin_disease_model.pth")

print("Training complete. Best validation accuracy:", best_val_acc)