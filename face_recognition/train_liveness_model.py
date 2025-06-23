import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.models import efficientnet_b0
import os
import time
from collections import Counter

# Define the EnhancedLivenessNet model
class EnhancedLivenessNet(nn.Module):
    def __init__(self):
        super(EnhancedLivenessNet, self).__init__()
        self.backbone = efficientnet_b0(weights='IMAGENET1K_V1')
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_features, 2)  # 2 classes: real (0), fake (1)
        )

    def forward(self, x):
        return self.backbone(x)

def check_dataset():
    """Check dataset structure and balance"""
    print("=" * 50)
    print("DATASET ANALYSIS")
    print("=" * 50)
    
    import shutil
    
    # Check and fix nested folder structure
    for split in ['real', 'fake']:
        path = f'dataset/{split}'
        if os.path.exists(path):
            files_moved = 0
            
            # Check for double nested structure: dataset/fake/dataset/fake/
            double_nested_path = f'dataset/{split}/dataset/{split}'
            if os.path.exists(double_nested_path):
                print(f"Found DOUBLE nested folder: {double_nested_path}")
                nested_files = [f for f in os.listdir(double_nested_path) if f.endswith('.jpg')]
                if nested_files:
                    print(f"Moving {len(nested_files)} files from double nested folder...")
                    for file in nested_files:
                        src = os.path.join(double_nested_path, file)
                        dst = os.path.join(path, file)
                        if not os.path.exists(dst):  # Don't overwrite existing files
                            shutil.move(src, dst)
                            files_moved += 1
                    print(f"Moved {files_moved} files to correct location")
                    
                    # Clean up empty nested folders
                    try:
                        os.rmdir(double_nested_path)
                        os.rmdir(f'dataset/{split}/dataset')
                        print(f"Removed empty nested folders")
                    except:
                        print("Some nested folders may still exist (not empty)")
            
            # Check for single nested structure: dataset/fake/dataset/
            elif os.path.exists(f'dataset/{split}/dataset'):
                nested_path = f'dataset/{split}/dataset'
                print(f"Found nested folder: {nested_path}")
                nested_files = [f for f in os.listdir(nested_path) if f.endswith('.jpg')]
                if nested_files:
                    print(f"Moving {len(nested_files)} files from nested folder...")
                    for file in nested_files:
                        src = os.path.join(nested_path, file)
                        dst = os.path.join(path, file)
                        if not os.path.exists(dst):  # Don't overwrite existing files
                            shutil.move(src, dst)
                            files_moved += 1
                    print(f"Moved {files_moved} files to correct location")
                    
                    # Remove empty nested folder
                    try:
                        os.rmdir(nested_path)
                        print(f"Removed empty nested folder")
                    except:
                        print("Nested folder may still exist (not empty)")
            
            # Now count files in correct location
            files = [f for f in os.listdir(path) if f.endswith('.jpg')]
            print(f"{split.upper()} images: {len(files)}")
            if len(files) > 0:
                print(f"  First file: {files[0]}")
                print(f"  Last file: {files[-1]}")
            else:
                print(f"  WARNING: No .jpg files found in {path}")
                # Check if there are still nested folders
                subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
                if subdirs:
                    print(f"  Found subdirectories: {subdirs}")
        else:
            print(f"ERROR: {path} folder not found!")
            return False
    
    # Load dataset to check class mapping
    try:
        temp_dataset = datasets.ImageFolder('dataset')
        print(f"\nClass mapping: {temp_dataset.class_to_idx}")
        print(f"Classes: {temp_dataset.classes}")
        print(f"Total images found: {len(temp_dataset)}")
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        return False
    
    print("=" * 50)
    return len(temp_dataset) > 0

def calculate_accuracy(outputs, labels):
    """Calculate accuracy"""
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return correct, total

def train_model():
    # Check dataset first
    if not check_dataset():
        return
    
    # Data transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load full dataset
    full_dataset = datasets.ImageFolder('dataset', transform=train_transform)
    
    # Split into train/validation (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Apply different transforms to validation set
    val_dataset.dataset.transform = val_transform
    
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    # Initialize model, loss, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = EnhancedLivenessNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)  # Lower LR + regularization
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)  # Learning rate decay
    
    # Training parameters
    num_epochs = 8
    best_val_acc = 0.0
    patience = 3
    patience_counter = 0
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print("=" * 70)
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 30)
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            correct, total = calculate_accuracy(outputs, labels)
            train_correct += correct
            train_total += total
            
            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                current_acc = 100 * train_correct / train_total
                current_loss = train_loss / (batch_idx + 1)
                print(f"  Batch {batch_idx+1}/{len(train_loader)}: "
                      f"Loss={current_loss:.4f}, Acc={current_acc:.2f}%")
        
        # Calculate training metrics
        train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                correct, total = calculate_accuracy(outputs, labels)
                val_correct += correct
                val_total += total
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch results
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch+1} Results ({epoch_time:.1f}s):")
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        print(f"  LR: {current_lr:.6f}")
        
        # Early stopping and model saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/enhanced_liveness_model.pth')
            print(f"  ✅ New best model saved! (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")
        
        print("=" * 70)
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered! Best validation accuracy: {best_val_acc:.2f}%")
            break
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved as: models/enhanced_liveness_model.pth")
    
    # Test loading the saved model
    print("\nTesting saved model...")
    test_model = EnhancedLivenessNet()
    test_model.load_state_dict(torch.load('models/enhanced_liveness_model.pth', map_location='cpu'))
    test_model.eval()
    print("✅ Model loads successfully!")
    
    return model

if __name__ == "__main__":
    try:
        train_model()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()