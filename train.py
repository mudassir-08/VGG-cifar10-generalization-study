import torch
import torch.nn as nn
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

from config import Confi
from model import VGGClassifier
from dataset import get_dataloaders

model = VGGClassifier()

batch_size = Confi.batch_size
validation_fraction = Confi.validation_fraction
image_size = Confi.image_size
seed = Confi.seed
num_workers = Confi.num_workers
weight_decay = Confi.decay_weight
device = Confi.device
epochs = Confi.epochs
lr_rate = Confi.lr_rate
grad_clip = Confi.grad_clip

train_loader, val_loader, test_loader = get_dataloaders(
    batch_size=batch_size,
    validation_fraction=validation_fraction,
    image_size=image_size,
    seed=seed,
    num_workers=num_workers
)

optimizer = AdamW(model.parameters(), lr=lr_rate, weight_decay=weight_decay)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
criterion = nn.CrossEntropyLoss()

best_acc = 0.0
counter = 0
patience = 10

history = {
    "train_loss": [],
    "val_loss": [],
    "val_acc": []
}

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("logs", exist_ok=True)

def validation(model, val_loader, device, criterion):
    model.eval()

    val_loss = 0.0
    correct = 0
    total = 0

    preds, targets = [], []

    with torch.no_grad():
        for features, labels in val_loader:

            features = features.to(device)
            labels = labels.to(device)

            output = model(features)
            loss = criterion(output, labels)

            val_loss += loss.item()

            pred = output.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

            preds.extend(pred.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    return val_loss / len(val_loader), correct / total

def train_model(model):

    model.to(device)
    scaler = GradScaler()

    global best_acc, counter

    for epoch in range(epochs):

        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")

        for features, labels in pbar:

            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=torch.cuda.is_available()):

                outputs = model(features)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / len(train_loader)
        val_loss, val_acc = validation(model, val_loader, device, criterion)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Acc: {val_acc:.4f}")

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch
        }, "checkpoints/latest_model.pth")

        if val_acc > best_acc:

            best_acc = val_acc
            counter = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_acc": best_acc
            }, "checkpoints/best_model.pth")

            print(f"Best model saved: {best_acc:.4f}")

        else:
            counter += 1

        if counter >= patience:
            print(" Early stopping triggered")
            break

    pd.DataFrame(history).to_csv("logs/history.csv", index=False)

    plt.figure()
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig("plots/loss.png")
    plt.close()

    plt.figure()
    plt.plot(history["val_acc"], label="val_acc")
    plt.legend()
    plt.title("Accuracy Curve")
    plt.savefig("plots/acc.png")
    plt.close()

    checkpoint = torch.load("checkpoints/best_model.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    model.eval()

    preds, targets = [], []

    with torch.no_grad():
        for x, y in test_loader:

            x, y = x.to(device), y.to(device)

            out = model(x)
            pred = out.argmax(1)

            preds.extend(pred.cpu().numpy())
            targets.extend(y.cpu().numpy())

    classes = [
        "airplane","automobile","bird","cat","deer",
        "dog","frog","horse","ship","truck"
    ]

    cm = confusion_matrix(targets, preds)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=classes,
                yticklabels=classes)

    plt.title("Confusion Matrix")
    plt.savefig("plots/confusion_matrix.png")
    plt.close()

    report = classification_report(targets, preds, target_names=classes)

    print(report)

    with open("logs/classification_report.txt", "w") as f:
        f.write(report)

train_model(model)