import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

class FERDataset(Dataset):
    CATEGORIES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    def __init__(self, data_dir, transform=None):
        self.transform = transform
        self.samples = []
        base_path = os.path.join(os.path.dirname(__file__), data_dir)
        print(f"Loading from: {os.path.abspath(base_path)}")
        for label, category in enumerate(self.CATEGORIES):
            folder = os.path.join(base_path, category)
            if not os.path.exists(folder):
                print(f"  Skipping '{category}'")
                continue
            for img_name in os.listdir(folder):
                self.samples.append((os.path.join(folder, img_name), label))
        print(f"  {len(self.samples)} images found\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (224, 224))
        img = np.stack([img, img, img], axis=2)
        img = transforms.functional.to_tensor(img)
        if self.transform:
            img = self.transform(img)
        return img, label

def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, 7)
    )
    return model

EPOCHS     = 40
BATCH_SIZE = 64
LR         = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}\n")

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
val_transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_set    = FERDataset("../train", transform=train_transform)
test_set     = FERDataset("../test",  transform=val_transform)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model     = build_model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

best_acc = 0.0
print("Phase 1: Training final layer only (epochs 1-15)...")

for epoch in range(1, EPOCHS + 1):

    if epoch == 16:
        print("\nPhase 2: Fine-tuning all layers...")
        for param in model.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    model.train()
    running_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    acc = correct / total * 100
    scheduler.step(100 - acc)
    print(f"Epoch {epoch:02d}/{EPOCHS} | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {acc:.2f}%")

    if acc > best_acc:
        best_acc = acc
        save_path = os.path.join(os.path.dirname(__file__), "emotion_model.pt")
        torch.save(model.state_dict(), save_path)
        print(f" New best saved ({acc:.2f}%)")

print(f"\nDone! Best accuracy: {best_acc:.2f}%")