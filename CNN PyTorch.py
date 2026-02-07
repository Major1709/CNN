import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

transforms_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
    transforms.ToTensor(),
])

transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder(root="/home/kevin/Downloads/shose/train", transform=transforms_train)
test_data = datasets.ImageFolder(root="/home/kevin/Downloads/shose/test", transform=transforms_test)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=2)
val_loader   = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=2)

num_classes = len(train_data.classes)

print(f"Number of classes: {num_classes}")

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            
        )
        
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
    
model = SimpleCNN(num_classes).to(device)
print(model)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(X)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>2f}  [{current:>5d}/{len(dataloader.dataset):>5d}]")
    avg_loss = total_loss / len(dataloader)
    print(f"Train Loss: {avg_loss:.4f}")

def test(dataloader, model, loss_fn):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = loss_fn(outputs, y)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / len(dataloader.dataset)
    print(f"Test Error:\n Accuracy: {100*accuracy:.1f}%, Avg loss: {avg_loss:.6f}\n")


for epoch in range(20):
    print(f"Epoch {epoch+1}\n-------------------------------")
    train(train_loader, model, loss_fn, optimizer)
    test(val_loader, model, loss_fn)

torch.save(model.state_dict(), "/home/kevin/Downloads/shos.pth")
model.load_state_dict(torch.load("/home/kevin/Downloads/shos.pth"))
from PIL import Image
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

img = Image.open("/home/kevin/Downloads/45e62.jpg")
img = transform(img).to(device)

model.eval()
with torch.no_grad():
    output = model(img.unsqueeze(0))
    pred = output.argmax(dim=1).item()

print("Prediction:", pred)
print(f"Number of classes: {train_data.classes[pred]}")
