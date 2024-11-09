import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F
# 固定随机种子
torch.manual_seed(7)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(7)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据加载
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.FashionMNIST('./F_MNIST_data', download=False, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testset = datasets.FashionMNIST('./F_MNIST_data', download=False, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

class NiN(nn.Module):
    def __init__(self, num_classes=10):
        super(NiN, self).__init__()
        # NiN Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 160, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(160, 96, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # NiN Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # NiN Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 10, kernel_size=1, stride=1, padding=0), # 10 is the number of classes
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)) # Global Average Pooling
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1) # Flatten the output
        return x

model = NiN(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.adam(model.parameters(), lr=0.001)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
# 计算并打印模型参数总数
total_params = count_parameters(model)
print(f'Total number of parameters: {total_params}')

start = time.time()
# 训练过程
epochs = 50
train_losses,train_accuracy,test_accuracy = [],[],[]
for epoch in range(epochs):
    running_loss = 0
    correct_count, total_count = 0, 0  # 用于计算训练准确率
    for images, labels in trainloader:
        images,labels = images.to(device),labels.to(device)
        optimizer.zero_grad()
        ps = model(images)
        loss = criterion(ps, labels)
        loss.backward()
        optimizer.step()
        # 计算本批次的损失
        running_loss += loss.item()
        # 计算本批次的准确率
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        correct_count += torch.sum(equals).item()
        total_count += len(labels)
    else:
        accuracy = 0
        with torch.no_grad():
            model.eval()
            for images, labels in testloader:
                images,labels = images.to(device),labels.to(device)
                ps = model(images)

                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))

        train_losses.append(running_loss / len(trainloader))
        train_accuracy.append(correct_count / total_count)
        test_accuracy.append(accuracy / len(testloader))
        print(f"Epoch {epoch+1}/{epochs}.. "
            f"Train loss: {running_loss/len(trainloader):.3f}.. "
            f"Train accuracy: {correct_count/total_count:.3f}.. "
            f"Test accuracy: {accuracy/len(testloader):.3f}")

end = time.time()
print(f"Time: {end-start:.2f}s")

epoch_range = range(1, epochs + 1)
plt.figure(figsize=(8, 6))
plt.plot(epoch_range, train_losses, 'r-', label='Training Loss')
plt.plot(epoch_range, test_accuracy, 'b-', label='Test Accuracy')
plt.plot(epoch_range, train_accuracy, 'g-', label='Train Accuracy')
plt.text(epochs/2, 0.5, f"Time: {end-start:.2f}s", ha='center', va='center', fontsize=12)
plt.xlabel('Epoch')
plt.xlim([1, epochs])
plt.ylim([0, 1])
plt.ylabel('Accuracy / Loss')
plt.title('Training Loss, Test Accuracy, and Train Accuracy vs. Epoch')
plt.legend()
plt.show()