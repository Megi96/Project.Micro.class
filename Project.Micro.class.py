import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

# Check if GPU is available (if not, will use CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define how we want to transform our images
# These transformations help the model learn better
transform = transforms.Compose([
    transforms.Resize((150, 150)),  # Resize all images to 150x150 pixels
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image colors
])

# Set the path to your dataset
data_path = r'C:\Users\C.S.T\Downloads\Micro_Organism'

# Check if the data directory exists
if not os.path.exists(data_path):
    raise ValueError(f"Data directory not found at {data_path}")

# Load the dataset
print("\nLoading dataset...")
dataset = torchvision.datasets.ImageFolder(root=data_path, transform=transform)

# Print information about the classes
print("\nClasses in the dataset:")
for class_idx, class_name in enumerate(dataset.classes):
    class_size = len([1 for _, label in dataset if label == class_idx])
    print(f"{class_name}: {class_size} images")

# Split dataset into training and validation sets (80% training, 20% validation)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

print(f"\nTotal images: {len(dataset)}")
print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")

# Create data loaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Define our CNN model
class SimpleMicroorganismCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleMicroorganismCNN, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # First layer: Input image -> 32 features(learns basic shapes)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Second layer: 32 -> 64 features(learns patterns)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Third layer: 64 -> 128 features(learns complex structures)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Classification layers - makes the final decision
        self.classifier = nn.Sequential(
            nn.Linear(128 * 18 * 18, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)  # Output size matches number of classes
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the features
        x = self.classifier(x)
        return x

# Create the model
num_classes = len(dataset.classes)  # This will be 8 for our dataset
print(f"\nCreating model with {num_classes} classes...")
model = SimpleMicroorganismCNN(num_classes).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training settings
num_epochs = 5
best_acc = 0
train_accs = []
val_accs = []

print("\nStarting training...")
print("=" * 50)

# Training loop
for epoch in range(num_epochs):
    # Training phase
    model.train()
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Print progress
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}]")
    
    # Calculate training accuracy
    train_acc = 100 * correct / total
    train_accs.append(train_acc)
    
    # Validation phase
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class accuracy
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1
    
    # Calculate validation accuracy
    val_acc = 100 * correct / total
    val_accs.append(val_acc)
    
    # Print epoch results
    print(f"\nEpoch {epoch+1}/{num_epochs}:")
    print(f"Training Accuracy: {train_acc:.2f}%")
    print(f"Validation Accuracy: {val_acc:.2f}%")
    
    # Print per-class accuracy
    print("\nPer-class accuracy:")
    for i in range(num_classes):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            print(f"{dataset.classes[i]}: {class_acc:.2f}%")
    
    print("=" * 50)
    
    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_micro_model.pth')
        print(f"New best model saved! Accuracy: {best_acc:.2f}%")

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(train_accs, label='Training Accuracy', marker='o')
plt.plot(val_accs, label='Validation Accuracy', marker='o')
plt.title('Training Progress')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.savefig('training_progress.png')
plt.show()

# Load best model and show predictions
print("\nLoading best model for predictions...")
model.load_state_dict(torch.load('best_micro_model.pth'))
model.eval()

# Display predictions
print("\nDisplaying sample predictions...")
with torch.no_grad():
    images, labels = next(iter(val_loader))
    images = images.to(device)
    outputs = model(images)
    _, preds = torch.max(outputs, 1)
    
    plt.figure(figsize=(15, 3))
    for i in range(min(5, len(images))):
        plt.subplot(1, 5, i+1)
        img = images[i].cpu().permute(1, 2, 0)
        img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
        img = torch.clamp(img, 0, 1)
        plt.imshow(img)
        true_label = dataset.classes[labels[i]]
        pred_label = dataset.classes[preds[i]]
        color = 'green' if true_label == pred_label else 'red'
        plt.title(f'True: {true_label}\nPred: {pred_label}', 
                 fontsize=8, 
                 color=color)
        plt.axis('off')
    plt.savefig('sample_predictions.png')
    plt.show()

print("\nTraining completed!")
print(f"Best validation accuracy: {best_acc:.2f}%")
print("\nResults have been saved:")
print("- Model: 'best_micro_model.pth'")
print("- Training progress plot: 'training_progress.png'")
print("- Sample predictions: 'sample_predictions.png'")