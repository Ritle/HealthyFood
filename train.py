# train.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os

# === Настройки ===
DATA_DIR = "data/food-101-split"
NUM_CLASSES = 101
BATCH_SIZE = 8
EPOCHS = 5  # для начала можно 5–10 эпох
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Трансформации ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# === Загрузка данных ===
train_ds = ImageFolder(f"{DATA_DIR}/train", transform=transform)
test_ds = ImageFolder(f"{DATA_DIR}/test", transform=transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# === Модель ===
model = models.efficientnet_b0(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model = model.to(DEVICE)

# === Оптимизатор и loss ===
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# === Обучение ===
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if i % 50 == 0:
            print(f"Epoch {epoch+1}, Batch {i}, Loss: {running_loss/50:.4f}")
            running_loss = 0.0

    # Сохраняем модель после каждой эпохи
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), f"models/food_model_epoch_{epoch+1}.pth")
    print(f"✅ Model saved: epoch {epoch+1}")

print("🎉 Обучение завершено!")