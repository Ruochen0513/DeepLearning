import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time

# 固定随机种子
torch.manual_seed(7)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(7)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据加载  #fashionMNIST数据集图像大小为: 28 x 28
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.FashionMNIST('./F_MNIST_data', download=False, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testset = datasets.FashionMNIST('./F_MNIST_data', download=False, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
# 定义网络结构
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 网络层定义
        ### 第一个卷积层设置 ###
        self.conv1 = nn.Conv2d(1, 6, 5) # 修改卷积核大小为5x5
        # self.bn1 = nn.BatchNorm2d(6)     # 添加BN层
        # self.conv1 = nn.Conv2d(1, 10, 5) #修改输出通道数为10,24
        # self.conv1 = nn.Conv2d(1,6,3,padding = 1) # 修改卷积核大小为3x3
        # self.conv1 = nn.Conv2d(1,6,7,padding = 1) # 修改卷积核大小为7x7
        ########################################
        ### 第二个卷积层设置 ###
        self.conv2 = nn.Conv2d(6, 16, 5)
        # self.bn2 = nn.BatchNorm2d(16)     # 添加BN层
        # self.conv2 = nn.Conv2d(10, 24, 5) # 修改输出通道数为10，24
        # self.conv2 = nn.Conv2d(6,16,3) # 修改卷积核大小为3x3
        # self.conv2 = nn.Conv2d(6,16,7) # 修改卷积核大小为7x7
        ########################################
        ### 池化层层设置 ###
        self.pool = nn.AvgPool2d(2, 2)   # 修改池化层为平均池化
        # self.pool = nn.MaxPool2d(2, 2) # 修改池化层为最大池化
        ### fc1层设置 ###
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        # self.fc1 = nn.Linear(24 * 4 * 4, 120) # 修改输出通道数为10，24
        # self.fc1 = nn.Linear(16 * 6 * 6, 120) # 修改全连接层输入大小匹配3x3卷积核
        # self.fc1 = nn.Linear(16 * 3 * 3, 120) # 修改全连接层输入大小匹配7x7卷积核
        ########################################
        self.fc2 = nn.Linear(120, 84)
        # self.bn3 = nn.BatchNorm1d(120)  # 添加BN层
        self.fc3 = nn.Linear(84, 10)
        # self.bn4 = nn.BatchNorm1d(84)   # 添加BN层
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    def forward(self, x):
        # 前向传播
        # x = self.pool(self.sigmoid(self.bn1(self.conv1(x))))  # 在conv1后加入BN
        # x = self.pool(self.sigmoid(self.bn2(self.conv2(x))))  # 在conv2后加入BN
        x = self.pool(self.sigmoid(self.conv1(x)))  
        x = self.pool(self.sigmoid(self.conv2(x)))  
        ### 全连接层输入设置 ###
        x = x.view(-1, 16 * 4 * 4)
        # x = x.view(-1, 24 * 4 * 4) # 修改输出通道数为10，24
        # x = x.view(-1, 16 * 6 * 6) # 修改全连接层输入大小匹配3x3卷积核
        # x = x.view(-1, 16 * 3 * 3) # 修改全连接层输入大小匹配7x7卷积核
        ########################################
        ### 激活函数输入设置 ###
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        # x = self.sigmoid(self.bn3(self.fc1(x)))  # 在fc1后加入BN
        # x = self.sigmoid(self.bn4(self.fc2(x)))  # 在fc2后加入BN
        # x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))  # 用ReLU激活函数
        ########################################
        x = self.fc3(x)
        return x

model = LeNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

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