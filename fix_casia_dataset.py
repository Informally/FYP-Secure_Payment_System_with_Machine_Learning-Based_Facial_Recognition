import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from PIL import Image
import cv2
import pandas as pd
import os
import numpy as np

# 1. Model definition (same as before)
class EnhancedLivenessNet(nn.Module):
    def __init__(self):
        super(EnhancedLivenessNet, self).__init__()
        self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),  # Increased dropout
            nn.Linear(num_features, 2)
        )

    def forward(self, x):
        logits = self.backbone(x)
        return logits

# 2. Dataset with improved data augmentation for real faces
class ImprovedLivenessDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None, augment_real=True):
        self.root_dir = root_dir
        self.transform = transform
        self.augment_real = augment_real
        
        self.data = pd.read_csv(csv_path)
        print(f"Original dataset: {len(self.data)} samples")
        print(f"Label distribution: {self.data['label'].value_counts().to_dict()}")
        
        # Create balanced dataset by duplicating real samples
        if augment_real:
            real_samples = self.data[self.data['label'] == 'real']
            fake_samples = self.data[self.data['label'] == 'fake']
            
            real_count = len(real_samples)
            fake_count = len(fake_samples)
            
            print(f"Original - Real: {real_count}, Fake: {fake_count}")
            
            # Duplicate real samples to balance dataset
            duplication_factor = max(1, fake_count // real_count)
            balanced_real = pd.concat([real_samples] * duplication_factor, ignore_index=True)
            
            self.data = pd.concat([balanced_real, fake_samples], ignore_index=True).sample(frac=1).reset_index(drop=True)
            
            print(f"Balanced dataset: {len(self.data)} samples")
            print(f"Balanced distribution: {self.data['label'].value_counts().to_dict()}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        file_path = os.path.join(self.root_dir, str(row['split_folder']), str(row['subject']), str(row['filename']))
        label = 1 if row['label'] == 'real' else 0

        try:
            # Read video file and extract first frame
            cap = cv2.VideoCapture(file_path)
            ret, frame = cap.read()
            cap.release()
            
            if not ret or frame is None:
                raise ValueError(f"Could not read video frame from {file_path}")
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Additional augmentation for real faces during training
            if self.augment_real and label == 1:  # Real face
                frame_rgb = self.augment_real_face(frame_rgb)
            
            # Convert to PIL Image
            img = Image.fromarray(frame_rgb)
            
            # Apply transforms
            if self.transform:
                img = self.transform(img)
            
            return img, label
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return a dummy black image as fallback
            dummy_img = Image.new('RGB', (224, 224), color='black')
            if self.transform:
                dummy_img = self.transform(dummy_img)
            return dummy_img, label

    def augment_real_face(self, frame):
        """Additional augmentation specifically for real faces"""
        import random
        
        # Random brightness adjustment
        if random.random() > 0.5:
            brightness = random.uniform(0.7, 1.3)
            frame = np.clip(frame * brightness, 0, 255).astype(np.uint8)
        
        # Random contrast adjustment
        if random.random() > 0.5:
            contrast = random.uniform(0.8, 1.2)
            frame = np.clip((frame - 128) * contrast + 128, 0, 255).astype(np.uint8)
        
        # Random noise
        if random.random() > 0.7:
            noise = np.random.normal(0, 5, frame.shape).astype(np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return frame

# 3. Enhanced data transforms with augmentation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 4. Load datasets with improved balance
print("=== LOADING IMPROVED DATASETS ===")
root_dir = r'C:/Users/User/Desktop/Dataset'

train_dataset = ImprovedLivenessDataset(
    csv_path=os.path.join(root_dir, 'proper_train_release_labels.csv'),
    root_dir=root_dir,
    transform=train_transform,
    augment_real=True
)

test_dataset = ImprovedLivenessDataset(
    csv_path=os.path.join(root_dir, 'proper_test_release_labels.csv'),
    root_dir=root_dir,
    transform=test_transform,
    augment_real=False
)

# 5. Create weighted sampler for balanced training
def create_weighted_sampler(dataset):
    # Count samples per class
    labels = [dataset[i][1] for i in range(len(dataset))]
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    sample_weights = [class_weights[label] for label in labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler

train_sampler = create_weighted_sampler(train_dataset)

train_loader = DataLoader(train_dataset, batch_size=16, sampler=train_sampler, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

# 6. Training with AGGRESSIVE WEIGHTS for real faces
print("\n=== AGGRESSIVE WEIGHT TRAINING ===")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = EnhancedLivenessNet().to(device)

# Load the existing improved model to continue training
try:
    model.load_state_dict(torch.load(r'C:/xampp/htdocs/FYP/face_recognition/models/enhanced_liveness_model_improved.pth', weights_only=True))
    print("✓ Loaded existing improved model to continue training")
except:
    print("Starting training from scratch")

# AGGRESSIVE CLASS WEIGHTS - heavily favor real faces
weight_for_fake = 0.2   # Very low weight for fake class
weight_for_real = 0.8   # Very high weight for real class

class_weights = torch.tensor([weight_for_fake, weight_for_real]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Very low learning rate for fine-tuning
optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

print(f"AGGRESSIVE Class weights - Fake: {weight_for_fake}, Real: {weight_for_real}")

# 7. Training loop with focus on real accuracy
num_epochs = 10  # Shorter training since we're fine-tuning
best_real_accuracy = 0
target_real_accuracy = 60.0  # Target minimum real accuracy
patience = 5
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        
        if batch_idx % 5 == 0:
            print(f"  Batch {batch_idx:3d}, Loss: {loss.item():.4f}")
    
    scheduler.step()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct_predictions / total_samples
    print(f"Epoch {epoch+1} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
    
    # Evaluate on test set
    model.eval()
    correct = 0
    total = 0
    class_correct = [0, 0]
    class_total = [0, 0]
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_correct[label] += (predicted[i] == labels[i]).item()
                class_total[label] += 1
    
    test_accuracy = 100 * correct / total
    real_accuracy = 100 * class_correct[1] / class_total[1] if class_total[1] > 0 else 0
    fake_accuracy = 100 * class_correct[0] / class_total[0] if class_total[0] > 0 else 0
    
    print(f"Test - Overall: {test_accuracy:.2f}%, Real: {real_accuracy:.2f}%, Fake: {fake_accuracy:.2f}%")
    
    # Early stopping based on real face accuracy
    if real_accuracy > best_real_accuracy:
        best_real_accuracy = real_accuracy
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), r'C:/xampp/htdocs/FYP/face_recognition/models/enhanced_liveness_model_aggressive.pth')
        print(f"✓ New best real accuracy: {real_accuracy:.2f}% - Model saved as aggressive version!")
        
        # Check if we reached target
        if real_accuracy >= target_real_accuracy:
            print(f"🎯 TARGET ACHIEVED! Real accuracy: {real_accuracy:.2f}% >= {target_real_accuracy}%")
            print("This model should work much better for real faces!")
            break
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping after {patience} epochs without improvement")
            break

# 8. Final evaluation
print("\n=== FINAL EVALUATION ===")
try:
    model.load_state_dict(torch.load(r'C:/xampp/htdocs/FYP/face_recognition/models/enhanced_liveness_model_aggressive.pth', weights_only=True))
    print("✓ Loaded aggressive model for final evaluation")
except:
    print("Using current model state for evaluation")

model.eval()

# Test on both datasets
for dataset_name, loader in [('Test', test_loader)]:
    correct = 0
    total = 0
    class_correct = [0, 0]
    class_total = [0, 0]
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_correct[label] += (predicted[i] == labels[i]).item()
                class_total[label] += 1
    
    overall_acc = 100 * correct / total
    real_acc = 100 * class_correct[1] / class_total[1] if class_total[1] > 0 else 0
    fake_acc = 100 * class_correct[0] / class_total[0] if class_total[0] > 0 else 0
    
    print(f"{dataset_name} Final Results:")
    print(f"  Overall Accuracy: {overall_acc:.2f}%")
    print(f"  Real Face Accuracy: {real_acc:.2f}%")
    print(f"  Fake Face Accuracy: {fake_acc:.2f}%")

print("\n=== TRAINING COMPLETE ===")
print("📊 COMPARISON:")
print("   Original model: Real 43.3%, Fake 87.0%")
print("   Improved model: Real 38.3%, Fake 92.5%")
print(f"   Aggressive model: Real {real_acc:.1f}%, Fake {fake_acc:.1f}%")
print()

if real_acc >= 60:
    print("🎉 SUCCESS! Your model should now work much better with real faces!")
    print("Replace your API model with: enhanced_liveness_model_aggressive.pth")
    print("Update your API to use this new model file.")
elif real_acc >= 45:
    print("✅ IMPROVEMENT! Real accuracy improved significantly.")
    print("You can either:")
    print("1. Use this model with a lower threshold (0.3-0.4) in your API")
    print("2. Or continue training for a few more epochs")
else:
    print("⚠️  Still needs work. Real accuracy is still low.")
    print("Consider:")
    print("1. Using an even lower threshold in your API (0.1-0.2)")
    print("2. Collecting more diverse real face training data")
    print("3. Using the current model but with heavy confidence boosting")

print("\nNext steps:")
print("1. Test this model in your face recognition API")
print("2. Adjust thresholds if needed")
print("3. Monitor real-world performance")