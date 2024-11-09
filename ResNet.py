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

# 定义一个基本的残差块
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# 定义ResNet网络
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# 实例化ResNet-18模型
def resnet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2],num_classes)

model = resnet18(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

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