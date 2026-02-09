import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F  # at the top if not already there

# ---- Your model from above ----
class CNN(nn.Module):
    def __init__(self, out_channels=32, pooling=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=5, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28 -> Conv(5) -> 24x24 -> Pool(2) -> 13x13
            nn.Conv2d(out_channels, out_channels=out_channels*2, kernel_size=5, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28 -> Conv(5) -> 24x24 -> Pool(2) -> 13x13
        )
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_channels * 2* 5 * 5, 10),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.mlp(x)
        return x

# ---- Utilities ----
def accuracy(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, running_acc, n_batches = 0.0, 0.0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        if isinstance(criterion, nn.CrossEntropyLoss):
            loss = criterion(logits, labels)
        elif isinstance(criterion, nn.MSELoss):
            targets = F.one_hot(labels, num_classes=logits.shape[-1]).float()
            loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc  += accuracy(logits, labels)
        n_batches    += 1

    return running_loss / n_batches, running_acc / n_batches

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, running_acc, n_batches = 0.0, 0.0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        if isinstance(criterion, nn.CrossEntropyLoss):
            loss = criterion(logits, labels)
        elif isinstance(criterion, nn.MSELoss):
            targets = F.one_hot(labels, num_classes=logits.shape[-1]).float()
            loss = criterion(logits, targets)
        running_loss += loss.item()
        running_acc  += accuracy(logits, labels)
        n_batches    += 1
    return running_loss / n_batches, running_acc / n_batches

def main():
    # Repro & device
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean/std
    ])

    data_root = os.path.join(os.getcwd(), "data")
    train_ds = datasets.MNIST(root=data_root, train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)

    batch_size = 64
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=256,      shuffle=False, num_workers=2, pin_memory=True)

    # Model, loss, optimizer
    model = CNN(out_channels=16).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=10e-2, weight_decay=1e-4)

    # Train
    epochs = 50
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc     = evaluate(model, test_loader, criterion, device)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "cnn_mnist_best.pth")

        print(f"Epoch {epoch:02d} | "
              f"train loss {train_loss:.4f} acc {train_acc*100:5.2f}% | "
              f"val loss {val_loss:.4f} acc {val_acc*100:5.2f}%")

    print(f"Best test acc: {best_acc*100:.2f}% (weights saved to cnn_mnist_best.pth)")

if __name__ == "__main__":
    main()
