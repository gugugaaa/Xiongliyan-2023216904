import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 1. 定义简单的 CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7) # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train():
    # 2. 数据准备
    transform = transforms.Compose([
        transforms.RandomAffine(
            degrees=10,          # 轻微旋转
            translate=(0.1, 0.1) # 轻微平移
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # MNIST 均值和方差
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 3. 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 4. 训练循环
    print(f"Start Training on {device}...")
    model.train()
    losses = []
    
    epochs = 3 # 简单任务3轮足够
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs} | Batch {batch_idx} | Loss: {loss.item():.4f}')
                losses.append(loss.item())

    # 5. 保存模型
    torch.save(model.state_dict(), "mnist_cnn.pth")
    print("Model saved as 'mnist_cnn.pth'")

    # 6. 可视化训练损失 (简明英文)
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("Training Loss Curve")
    plt.xlabel("Steps (x100)")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("training_loss.png")
    plt.show()

if __name__ == '__main__':
    train()