import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import cv2
import pandas as pd
import os

# 1. Model definition (must match your system)
class EnhancedLivenessNet(nn.Module):
    def __init__(self):
        super(EnhancedLivenessNet, self).__init__()
        self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_features, 2)
        )

    def forward(self, x):
        logits = self.backbone(x)
        return logits  # Use logits for nn.CrossEntropyLoss

# 2. Dataset loader (frame extraction from video or image)
class LivenessDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):  # Fixed indentation - this method is now properly inside the class
        attempts = 0
        original_idx = idx
        
        while attempts < 10:
            row = self.data.iloc[idx]
            file_path = os.path.join(self.root_dir, str(row['subject']), row['filename'])
            label = 1 if row['label'] == 'real' else 0

            # Check if file exists
            if not os.path.exists(file_path):
                print(f"Warning: File does not exist: {file_path}")
                idx = (idx + 1) % len(self)
                attempts += 1
                continue

            # Load image or first frame of video
            if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                try:
                    img = Image.open(file_path).convert('RGB')
                    if img.size[0] < 32 or img.size[1] < 32:  # Check minimum size
                        print(f"Warning: Image too small: {file_path}")
                        idx = (idx + 1) % len(self)
                        attempts += 1
                        continue
                    if self.transform:
                        img = self.transform(img)
                    return img, label
                except Exception as e:
                    print(f"Warning: Could not read image {file_path}: {e}")
            else:
                # Enhanced video reading with multiple backends
                try:
                    # Try different backends for video reading
                    backends = [cv2.CAP_FFMPEG, cv2.CAP_ANY]
                    frame_read = False
                    
                    for backend in backends:
                        cap = cv2.VideoCapture(file_path, backend)
                        if cap.isOpened():
                            ret, frame = cap.read()
                            cap.release()
                            if ret and frame is not None:
                                # Check frame dimensions
                                if frame.shape[0] < 32 or frame.shape[1] < 32:
                                    print(f"Warning: Video frame too small: {file_path}")
                                    break
                                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                                if self.transform:
                                    img = self.transform(img)
                                return img, label
                        else:
                            cap.release()
                    
                    if not frame_read:
                        print(f"Warning: Could not read video {file_path} with any backend")
                        
                except Exception as e:
                    print(f"Warning: Error processing video {file_path}: {e}")
            
            # Try next sample
            idx = (idx + 1) % len(self)
            attempts += 1
            
            # If we've tried all samples, break to avoid infinite loop
            if idx == original_idx:
                break
        
        # If we can't find any valid samples, create a dummy sample
        print(f"Warning: Creating dummy sample after {attempts} failed attempts")
        dummy_img = Image.new('RGB', (224, 224), color='black')
        if self.transform:
            dummy_img = self.transform(dummy_img)
        return dummy_img, 0

# 3. Transforms (must match your inference code)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 4. Prepare data
print("Loading and analyzing dataset...")

# First, let's check the CSV files
try:
    train_csv = pd.read_csv(r'C:/Users/User/Desktop/Dataset/merged_train_labels.csv')
    print(f"Train CSV shape: {train_csv.shape}")
    print(f"Train CSV columns: {train_csv.columns.tolist()}")
    print(f"First few rows of train CSV:")
    print(train_csv.head())
    print(f"Label distribution in train: {train_csv['label'].value_counts()}")
    
    # Check file extensions in the dataset
    if 'filename' in train_csv.columns:
        extensions = train_csv['filename'].str.split('.').str[-1].value_counts()
        print(f"File extensions in train dataset: {extensions}")
    
except Exception as e:
    print(f"Error reading train CSV: {e}")

try:
    test_csv = pd.read_csv(r'C:/Users/User/Desktop/Dataset/merged_test_labels.csv')
    print(f"\nTest CSV shape: {test_csv.shape}")
    print(f"Label distribution in test: {test_csv['label'].value_counts()}")
except Exception as e:
    print(f"Error reading test CSV: {e}")

# Check if some sample files exist
print("\nChecking sample files...")
sample_files = [
    r'C:/Users/User/Desktop/Dataset/18/1.avi',
    r'C:/Users/User/Desktop/Dataset/19/1.avi'
]

for file_path in sample_files:
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path)
        print(f"File exists: {file_path} (Size: {file_size} bytes)")
        
        # Try to get video info
        cap = cv2.VideoCapture(file_path)
        if cap.isOpened():
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  Video info: {width}x{height}, {frame_count} frames, {fps} FPS")
        else:
            print(f"  Cannot open video: {file_path}")
        cap.release()
    else:
        print(f"File missing: {file_path}")

train_dataset = LivenessDataset(
    csv_path=r'C:/Users/User/Desktop/Dataset/merged_train_labels.csv',
    root_dir=r'C:/Users/User/Desktop/Dataset',
    transform=transform
)
print(f"\nTrain dataset created with {len(train_dataset)} samples")
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = LivenessDataset(
    csv_path=r'C:/Users/User/Desktop/Dataset/merged_test_labels.csv',
    root_dir=r'C:/Users/User/Desktop/Dataset',
    transform=transform
)
print(f"Test dataset created with {len(test_dataset)} samples")
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 5. Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EnhancedLivenessNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):  # Adjust epochs as needed
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
    print(f"Epoch {epoch+1} done. Loss: {running_loss/len(train_loader):.4f}")

# 6. Save the model (for your system)
torch.save(model.state_dict(), r'C:/xampp/htdocs/FYP/face_recognition/models/enhanced_liveness_model.pth')
print("Model saved!")

# 7. Evaluate on test set
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Test Accuracy: {100 * correct / total:.2f}%')