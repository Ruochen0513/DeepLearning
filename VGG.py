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

# 数据加载
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.FashionMNIST('./F_MNIST_data', download=False, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testset = datasets.FashionMNIST('./F_MNIST_data', download=False, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
# 定义网络结构
import torch
import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self, num_classes):
        super(VGG16, self).__init__()
        # 定义VGG16的卷积层部分
        self.features = nn.Sequential(
            # 两个3x3卷积层，输出通道数为64，后跟一个最大池化层
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 两个3x3卷积层，输出通道数为128，后跟一个最大池化层
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 受fashionMNIST数据集的限制，这里只使用了4个卷积层，两个最大池化
        )
        # 定义VGG16的全连接层部分
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 4096),  # 输入大小取决于输入图像的大小和前面的卷积层
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 展平张量用于全连接层
        x = self.classifier(x)
        return x

model = VGG16(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

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