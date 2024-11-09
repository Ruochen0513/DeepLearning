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

# 定义网络结构
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        # 第一条分支：1x1卷积
        self.branch1 = nn.Conv2d(in_channels, ch1x1, kernel_size=1)

        # 第二条分支：1x1卷积 + 3x3卷积
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1),  # 1x1卷积
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)  # 3x3卷积，保持输入和输出尺寸一致
        )

        # 第三条分支：1x1卷积 + 5x5卷积
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1),  # 1x1卷积
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)  # 5x5卷积，保持输入和输出尺寸一致
        )

        # 第四条分支：3x3最大池化 + 1x1卷积
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),  # 3x3最大池化，保持输入和输出尺寸一致
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)  # 1x1卷积
        )

    def forward(self, x):
        # 将输入分别传入四个分支
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        # 将四个分支的输出在通道维度上进行拼接
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class GoogLeNet(nn.Module):
    def __init__(self, num_classes):
        super(GoogLeNet, self).__init__()

        # 第一层卷积层
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 第二层卷积层
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 第三层Inception模块
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 第四层Inception模块
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 第五层Inception模块
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        # 平均池化层，Dropout层，全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        # 通过第一层卷积和池化
        x = self.maxpool1(F.relu(self.conv1(x)))

        # 通过第二层卷积和池化
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.maxpool2(x)

        # 通过第三层Inception模块
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        # 通过第四层Inception模块
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        # 通过第五层Inception模块
        x = self.inception5a(x)
        x = self.inception5b(x)

        # 通过平均池化、Dropout和全连接层
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.dropout(x)
        x = self.fc(x)
        return x

model = GoogLeNet(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003)

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