import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import copy
from datetime import datetime

# Dataset directory
DATA_DIR = 'data'
CLASS_NAMES = ['fighting', 'running', 'sitting', 'talking', 'walking']
NUM_CLASSES = len(CLASS_NAMES)

# Training parameters
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.0005
IMAGE_SIZE = 224

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data augmentation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_datasets():
    print("Loading datasets...")
    
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=val_transform)
    
    # Get class counts
    class_counts = Counter([label for _, label in train_dataset.samples])
    print("Class distribution in training set:", class_counts)
    
    # Calculate class weights
    total_samples = sum(class_counts.values())
    class_weights = {cls: total_samples / (len(class_counts) * count) for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for _, label in train_dataset.samples]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    }
    
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    class_names = train_dataset.classes
    
    print(f"Class names: {class_names}")
    print(f"Train images: {dataset_sizes['train']}")
    print(f"Validation images: {dataset_sizes['val']}")
    
    return dataloaders, dataset_sizes, class_names

class EmotionRecognitionModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(EmotionRecognitionModel, self).__init__()
        # Use ConvNeXt as base model
        self.base_model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1)
        
        # Freeze initial layers
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Unfreeze last few layers
        for param in self.base_model.features[-5:].parameters():
            param.requires_grad = True
            
        # Modify classifier head
        in_features = self.base_model.classifier[2].in_features
        self.base_model.classifier[2] = nn.Sequential(
            nn.LayerNorm(in_features, eps=1e-6),
            nn.Flatten(1),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        return self.base_model(x)

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs):
    since = datetime.now()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    all_labels = []
    all_preds = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            current_labels = []
            current_preds = []

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                current_labels.extend(labels.cpu().numpy())
                current_preds.extend(preds.cpu().numpy())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val':
                all_labels.extend(current_labels)
                all_preds.extend(current_preds)
                
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), 'best_model.pth')
                    print("Saved new best model")
                
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(epoch_acc)
                elif isinstance(scheduler, optim.lr_scheduler.CosineAnnealingLR):
                    scheduler.step()

        print()

    time_elapsed = datetime.now() - since
    print(f'Training complete in {time_elapsed}')
    print(f'Best val Acc: {best_acc:.4f}')
    
    # Save training curves
    plot_training_history(history)
    
    # Save classification report and confusion matrix
    save_classification_metrics(all_labels, all_preds)
    
    model.load_state_dict(best_model_wts)
    return model

def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def save_classification_metrics(true_labels, pred_labels):
    print("\nClassification Report:")
    report = classification_report(true_labels, pred_labels, target_names=CLASS_NAMES, output_dict=True)
    print(classification_report(true_labels, pred_labels, target_names=CLASS_NAMES))
    
    # Save report to file
    with open('classification_report.txt', 'w') as f:
        f.write(classification_report(true_labels, pred_labels, target_names=CLASS_NAMES))
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Per-class accuracy plot
    class_acc = {}
    for i, class_name in enumerate(CLASS_NAMES):
        mask = np.array(true_labels) == i
        class_acc[class_name] = np.mean(np.array(pred_labels)[mask] == i)
    
    plt.figure(figsize=(10, 5))
    plt.bar(class_acc.keys(), class_acc.values())
    plt.title('Per-class Accuracy')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.savefig('per_class_accuracy.png')
    plt.close()

def main():
    print("Starting emotion recognition training...")
    print(f"Classes: {CLASS_NAMES}")
    
    # Load data
    dataloaders, dataset_sizes, class_names = load_datasets()
    
    # Initialize model
    model = EmotionRecognitionModel(num_classes=NUM_CLASSES)
    model = model.to(device)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.AdamW([
        {'params': [p for p in model.parameters() if p.requires_grad]},
    ], lr=LEARNING_RATE, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    
    # Train the model
    model = train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, NUM_EPOCHS)
    
    # Save the final model
    torch.save(model.state_dict(), 'final_model.pth')
    print("Model saved as 'final_model.pth'")

if __name__ == '__main__':
    main()
