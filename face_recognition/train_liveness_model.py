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

# 2. Dataset loader (frame extraction from video)
class CasiaFASDDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        video_path = os.path.join(self.root_dir, str(row['subject']), row['filename'])
        label = 1 if row['label'] == 'real' else 0

        # Extract the first frame (or random frame) from the video
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Could not read video {video_path}")

        # Convert to PIL Image for transforms
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame)
        if self.transform:
            pil_img = self.transform(pil_img)
        return pil_img, label

# 3. Transforms (must match your inference code)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 4. Prepare data
train_dataset = CasiaFASDDataset(
    csv_path=r'C:/Users/User/Desktop/Dataset/train_labels.csv',
    root_dir=r'C:/Users/User/Desktop/Dataset/train_release',
    transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# 5. Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EnhancedLivenessNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):  # Adjust epochs as needed
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} done.")

# 6. Save the model (for your system)
torch.save(model.state_dict(), r'C:/xampp/htdocs/FYP/face_recognition/models/enhanced_liveness_model.pth')