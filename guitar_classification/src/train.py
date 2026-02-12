import torch
from torch.utils.data import DataLoader
from dataset import GuitarStyleDataset
from model import CNNLSTM
import os



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 2. 프로젝트 루트를 기준으로 경로 재설정
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
VAL_DIR = os.path.join(BASE_DIR, "dataset", "val")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

# 3. 폴더 생성 (작업 디렉토리와 상관없이 항상 프로젝트 내부에 생성됨)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3

# Dataset / DataLoader
train_dataset = GuitarStyleDataset(TRAIN_DIR)
val_dataset = GuitarStyleDataset(VAL_DIR)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model
model = CNNLSTM().to(DEVICE)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Train loop
best_acc = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = correct / total * 100
    print(f"[{epoch+1}/{EPOCHS}] loss: {avg_loss:.4f}, val acc: {acc:.2f}%")

    # --- 모델 저장 ---
    # best_model.pt
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pt"))

# 마지막 epoch 모델 저장
torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "last_model.pt"))