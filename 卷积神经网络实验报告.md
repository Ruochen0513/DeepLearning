## <center>卷积神经网络实验报告
#### 一、实验目的
- 了解卷积神经网络(CNN)的网络模型
- 使用Fashion-MNIST数据集训练LeNet网络并分别画出损失函数、训练误差、测试误差随迭代epoch的变化曲线，直观展示LeNet的性能
- 掌握以下更改对于卷积神经网络性能影响：<br/>
  1.更改激活函数从sigmoid到ReLU<br/>
  2.将sigmoid与batch normalization结合使用<br/>
  3.调整卷核的大小<br/>
  4.调整输出通道数量<br/>
  5.将average polling替换为max polling<br/>
- 试用其他不同的卷积神经网络如VGG、GoogLeNet、NIN、ResNet并观察其效果
#### 二、实验过程
##### 1.网络构建
&emsp;&emsp;LeNet网络由Yann Lecun等人提出，是一种经典的卷积神经网络，是现代卷积神经网络的起源之一。LeNet具有一个输入层，两个卷积层，两个池化层，三个全连接层（其中最后一个全连接层为输出层）。
```python
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
```
&emsp;&emsp;在以上代码中我们构建了一个输入层，两个卷积层，两个池化层，三个全连接层（其中最后一个全连接层为输出层）。</br>
&emsp;&emsp;第一个卷积层将1个输入通道（通常是灰度图像）转换成6个输出通道，使用5x5的卷积核。第二个卷积层则将6个输入通道转换成16个输出通道，同样使用5x5的卷积核。</br>
&emsp;&emsp;在每层卷积并激活之后对其进行平均池化，池化层使用2x2的窗口，并且步长也是2。</br>
&emsp;&emsp;对于三个全连接层，它们分别有120,84,和10个神经元。这里的10是最后的输出类别的数量。</br>
&emsp;&emsp;每层之后的激活函数选用sigmoid函数。具体的LeNet网络结构如下图所示：</br>
![alt text](cnn1.png)
&emsp;&emsp;在上述的模型代码中为实验提供更多参数选择：
- 选择激活函数为sigmoid或ReLu
- 选择是否在每层之后添加batch normalization层
- 修改卷积核大小为3x3、5x5、7x7
- 修改两个卷积层输出通道的大小分别为6，16或10，24
- 修改池化层为平均池化或最大池化</br>

&emsp;&emsp;实际实验中可根据需要查看不同参数选择的效果。
##### 2.网络训练
```python
model = LeNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
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
```
&emsp;&emsp;在模型训练中选用损失函数为交叉熵损失函数，优化算法为随机梯度下降(SGD)，其中学习率(lr)=0.1，动量(momentum)=0.9，迭代次数为50次。在每次迭代之后输出当前的训练损失、训练准确率和测试准确率。
#### 三、实验要求及结果展示
&emsp;&emsp;LeNet默认参数设置：卷积核大小为5x5，两个卷积层输出通道个数分别为6和16，采用平均池化，激活函数选用sigmoid，未添加BN层。
##### 1.使用LeNet，分别画出损失函数、训练误差、测试误差随迭代epoch的变化曲线
&emsp;&emsp;在默认参数设置下固定pytorch随机种子为7，此时参数总量为44426，训练fashionMNIST数据集迭代50次，结果如下：
![alt text](<LeNet 50  0.91.png>)
&emsp;&emsp;在迭代50次之后测试准确率为91%左右，可见LeNet对于fashionMNIST数据集实现了有效分类，但存在一定过拟合现象。
##### 2.将sigmoid替换成ReLU，效果如何？
```python
    def forward(self, x):
        # 前向传播
        x = self.pool(self.relu(self.conv1(x)))  
        x = self.pool(self.relu(self.conv2(x)))  
        ### 全连接层输入设置 ###
        x = x.view(-1, 16 * 4 * 4)  
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))  # 用ReLU激活函数
        x = self.fc3(x)
        return x
```
&emsp;&emsp;将前向传播函数部分各层激活函数全改为ReLu激活函数，迭代50次后结果如下：
![alt text](<lenet relu 50 0.897.png>)
&emsp;&emsp;在迭代50次之后测试准确率为89.7%左右，相比于上面采用sigmoid激活函数的情况，使用ReLu之后测试准确率更快地到达一个比较高的值，因此ReLu相比于sigmoid具有更高的计算效率，而且在深度网络中使用ReLu激活函数可以缓解梯度消失问题。
##### 3.将sigmoid与batch normalization结合使用，效果如何？
```python
# 定义网络结构
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 网络层定义
        self.conv1 = nn.Conv2d(1, 6, 5) # 修改卷积核大小为5x5
        self.bn1 = nn.BatchNorm2d(6)     # 添加BN层
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)     # 添加BN层
        self.pool = nn.AvgPool2d(2, 2)   # 修改池化层为平均池化
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.bn3 = nn.BatchNorm1d(120)  # 添加BN层
        self.fc3 = nn.Linear(84, 10)
        self.bn4 = nn.BatchNorm1d(84)   # 添加BN层
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    def forward(self, x):
        # 前向传播
        x = self.pool(self.sigmoid(self.bn1(self.conv1(x))))  # 在conv1后加入BN
        x = self.pool(self.sigmoid(self.bn2(self.conv2(x))))  # 在conv2后加入BN 
        x = x.view(-1, 16 * 4 * 4)
        x = self.sigmoid(self.bn3(self.fc1(x)))  # 在fc1后加入BN
        x = self.sigmoid(self.bn4(self.fc2(x)))  # 在fc2后加入BN
        x = self.fc3(x)
        return x
```
&emsp;&emsp;在默认网络模型中每层结束之后添加一个batch normalization层，迭代50次之后结果如下：
![alt text](<lenet 50 batchnormalization 0.877.png>)
|      | 未加BN层     | 添加BN层|
| -------- | -------- | --------|
|训练准确率  | 94.8% | 87.9%|
| 测试准确率 | 91.0% |87.7%|
| 训练时间/s | 1846.42 |1271.06|

&emsp;&emsp;在经过50次迭代之后最终测试准确率为87.7%左右，相比于默认不使用BN层的情况，此时训练速度更快，测试准确率很快达到一个比较高的值；而且测试准确率和训练准确率更加吻合了，过拟合情况大大减小。</br>
&emsp;&emsp;由此可见在使用了BN层之后计算效率大大提升，而且大大提高了网络的泛化能力。
##### 4.调整卷核的大小，效果如何？
&emsp;&emsp;**将卷积核大小从5x5改为3x3：**
```python
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 网络层定义
        self.conv1 = nn.Conv2d(1,6,3,padding = 1) # 修改卷积核大小为3x3
        self.conv2 = nn.Conv2d(6,16,3) # 修改卷积核大小为3x3
        self.pool = nn.AvgPool2d(2, 2)   # 修改池化层为平均池化
        self.fc1 = nn.Linear(16 * 6 * 6, 120) # 修改全连接层输入大小匹配3x3卷积核
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
```
&emsp;&emsp;fashionMNIST数据集图像为28x28，所以在选用3x3卷积核之后为了使第一个卷积层输出大小为整数，我们添加了一个大小为1的padding，图像大小经过前向网络过程如下：
- 经过第一个卷积层：$ \frac{28-3+2}{1}+1 = 28 $
- 经过第一个平均池化层：$ \frac{28}{2} = 14 $
- 经过第二个卷积层：$ \frac{14-3}{1}+1 = 12 $
- 经过第二个平均池化层：$ \frac{12}{2} = 6 $ </br>

&emsp;&emsp;因此第一个全连接层的输入为$6\times6$。在迭代50次之后结果如下：
![](<lenet 50 3x3卷积核 0.909.png>)
|      | $3\times3$卷积核     | $5\times5$卷积核|
| -------- | -------- | --------|
|训练准确率  | 93.8% | 94.8%|
| 测试准确率 | 90.9% |91.0%|
| 训练时间/s | 1638.83 |1846.42|

&emsp;&emsp;迭代50次之后测试集准确率为90.9%左右，相比于$5\times5$卷积核来说，采用$3\times3$卷积核会使训练时间有所减少，过拟合现象有所缓解。</br>
&emsp;&emsp;较小的卷积核意味着每个神经元的感受野更小，因此它们可能只能捕捉到输入数据中的局部特征，通过堆叠多个较小的卷积核能够构建非常强大的特征提取器。更小的卷积核包含更少的参数，这意味着整个网络的参数量会减少，有助于减少过拟合的风险，并加快训练速度。</br>

&emsp;&emsp;**将卷积核大小从5x5改为7x7：**
```python
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 网络层定义
        self.conv1 = nn.Conv2d(1,6,7,padding = 1) # 修改卷积核大小为7x7
        self.conv2 = nn.Conv2d(6,16,7) # 修改卷积核大小为7x7
        self.pool = nn.AvgPool2d(2, 2)   # 修改池化层为平均池化
        self.fc1 = nn.Linear(16 * 6 * 6, 120) # 修改全连接层输入大小匹配7x7卷积核
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
```
&emsp;&emsp;fashionMNIST数据集图像为28x28，所以在选用3x3卷积核之后为了使第一个卷积层输出大小为整数，我们添加了一个大小为1的padding，图像大小经过前向网络过程如下：
- 经过第一个卷积层：$ \frac{28-7+2}{1}+1 = 24 $
- 经过第一个平均池化层：$ \frac{24}{2} = 12 $
- 经过第二个卷积层：$ \frac{12-7}{1}+1 = 6 $
- 经过第二个平均池化层：$ \frac{6}{2} = 3 $ </br>

&emsp;&emsp;因此第一个全连接层的输入为$3\times3$。在迭代50次之后结果如下：
![alt text](<lenet 50 7x7卷积核 0.897.png>)
&emsp;&emsp;迭代50次之后测试集准确率为89.7%左右，相比于$5\times5$卷积核来说，采用$7\times7$卷积核会使训练时间有所增加。</br>  
|      | $7\times7$卷积核     | $5\times5$卷积核|
| -------- | -------- | --------|
|训练准确率  | 92.5% | 94.8%|
| 测试准确率 | 89.7% |91.0%|
| 训练时间/s | 2263.70 |1846.42|

&emsp;&emsp;更大的卷积核包含更多的参数，这意味着整个网络的参数量会增加，这不仅增加了模型的计算复杂度，还可能增加训练时间和所需的内存。$7\times7$的卷积核具有更大的感受野，能够一次性捕获更大范围的信息。这对于识别较大尺度的特征或模式可能是有利的，但它也可能导致丢失一些细粒度的细节信息。</br>
##### 5.调整输出通道数量，效果如何？
```python
# 定义网络结构
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 网络层定义
        self.conv1 = nn.Conv2d(1, 10, 5) #修改输出通道数为10,24
        self.conv2 = nn.Conv2d(10, 24, 5) # 修改输出通道数为10，24
        self.pool = nn.AvgPool2d(2, 2)   # 修改池化层为平均池化
        self.fc1 = nn.Linear(24 * 4 * 4, 120) # 修改输出通道数为10，24
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
```
&emsp;&emsp;如上所示，将第一个卷积层的输出通道从6改为10，将第二个卷积层的输出通道从16改为24，在经过50次迭代之后结果如下：
![alt text](<lenet 50 更改通道数 0.903.png>)
|      | 输出通道量：6，16     | 输出通道量：10，24|
| -------- | -------- | --------|
|训练准确率  | 95.0% | 94.8%|
| 测试准确率 | 90.3% |91.0%|
| 训练时间/s | 2025.92 |1846.42|

&emsp;&emsp;在经过50次迭代之后测试准确率为90.3%左右，训练时间有所增长，测试准确率变化不大。</br>
&emsp;&emsp;每个卷积层的输出通道数量增加，意味着模型的参数数量会相应增加。随着参数数量的增加，每层的计算量也会增加，这将导致训练和推理的时间延长。同时，内存消耗也会增加，因为需要存储更多的权重和激活值。更多的输出通道意味着网络可以学习更多种类的特征，从而增加模型的表达能力和复杂性。这可以提高模型对复杂模式的捕捉能力，但也可能导致过拟合的风险增加，特别是在训练数据集较小时。
##### 6.将average polling替换为max polling，效果如何？
```python
    self.pool = nn.MaxPool2d(2, 2) # 修改池化层为最大池化
```
&emsp;&emsp;将网络模型中的池化函数改为最大池化，在经过50次迭代之后结果如下：
![alt text](<lenet 50 maxpool 0.895.png>)
|      | max polling   | average polling|
| -------- | -------- | --------|
|训练准确率  | 94.6% | 94.8%|
| 测试准确率 | 89.5% |91.0%|
| 训练时间/s | 1072.36 |1846.42|

&emsp;&emsp;在经过50次迭代之后，测试准确率为89.5%左右。相比于平均池化，测试准确率变化不大、计算时间有所下降。</br>
&emsp;&emsp;两种池化方式的计算复杂度相似，但从实现角度来看，最大池化的操作通常是简单的比较操作，而平均池化需要进行加法运算后除以区域大小，理论上最大池化可能略微快一些。</br>
&emsp;&emsp;最大池化可以提供某种程度的平移不变性，这意味着即使对象在图像中稍微移动了一些位置，最大池化仍然可能选择相同的特征点，因此网络对于小的位移不太敏感。</br>
&emsp;&emsp;在反向传播过程中，最大池化只有那些被选中的最大值对应的输入节点会收到梯度，其他节点的梯度为零。这种稀疏的梯度更新机制可能有助于减少过拟合。
#### 四、附加题
&emsp;&emsp;**由于我本人电脑GPU为NVIDA GeForce MX350，其对于较深的神经网络训练较慢，所以对于附加题的VGG、NIN、GoogLeNet、ResNet皆使用Google Colab T4 GPU环境来训练。**
##### 1.根据自身算力，尝试合适大小的VGG，效果如何？
&emsp;&emsp;2014年，牛津大学提出了另一种深度卷积网络VGG-Net，它相比于AlexNet有更小的卷积核和更深的层级。AlexNet前面几层用了11×11和5×5的卷积核以在图像上获取更大的感受野，而VGG采用更小的卷积核与更深的网络提升参数效率。
![alt text](image.png)
&emsp;&emsp;如上图所示，作者在VGGNet的实验中只用了两种卷积核大小：1x1和3x3。作者认为两个3x3的卷积堆叠获得的感受野大小，相当一个5x5的卷积；而3个3x3卷积的堆叠获取到的感受野相当于一个7x7的卷积。同时，VGG相比于AlexNet采用更深的网络结构，其具体结构如下：
![alt text](image-1.png)
&emsp;&emsp;受计算性能的影响，我在实际训练时减少了VGG的网络层数，只保留了两组各两个$3\times3$的卷积层，如下图所示：
```python
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
```
&emsp;&emsp;卷积层的输入和最后全连接层的输出都与FashionMNIST数据集作了匹配，之后的优化器我们使用了SGD，学习率为0.1。在经过50次迭代后结果如下：
![alt text](<VGG 50 0.933.png>)
|      | VGG16   | LeNet|
| -------- | -------- | --------|
|训练准确率  | 100% | 94.8%|
| 测试准确率 | 93.3% |91.0%|
| 训练时间/s | 1409.13 |1846.42|

&emsp;&emsp;由于附加题的训练环境与LeNet训练环境不同，所以这里的训练时间仅做参考，但相比于LeNet，VGG16的测试准确率和训练准确率有所提升，可见VGG对于图像分类的有效性。但同时由训练曲线也可见VGG存在一定的过拟合现象。
##### 2.根据自身算力，尝试合适大小的NIN，效果如何？
&emsp;&emsp;Network in Network (NIN) 是一种卷积神经网络（CNN）架构，由 Min Lin 等人于 2013 年提出。NIN 的核心思想是通过“网络嵌套网络”来提升特征表达能力，具体方法是用小型的多层感知器（MLP）替代传统 CNN 的卷积层。NIN 在网络结构上带来了两个重要创新：MLP卷积和全局平均池化。</br>
&emsp;&emsp;NiN中，在卷积层中使用了一种微型网络(micro network)提高卷积层的抽象能力，这里使用多层感知器(MLP)作为micro network，因为这个MLP卫星网络是位于卷积网络之中的，因此该模型被命名为“network in network”  ，下图对比了普通的线性卷积层和多层感知器卷积层(MlpConv Layer)。
![alt text](image-2.png)
&emsp;&emsp;线性卷积层和MLPConv都是将局部感受野(local receptive field)映射到输出特征向量。MLPConv核使用带非线性激活函数的MLP，跟传统的CNN一样，MLP在各个局部感受野中共享参数的，滑动MLP核可以最终得到输出特征图。NIN通过多个MLPConv的堆叠得到。完整的NiN网络结构如下图所示。
![alt text](image-3.png)
&emsp;&emsp;传统的CNN模型先是使用堆叠的卷积层提取特征，输入全连接层(FC)进行分类。这种结构沿袭自LeNet5，把卷积层作为特征抽取器，全连接层作为分类器。但是FC层参数数量太多，很容易过拟合，会影响模型的泛化性能。因此需要用Dropout增加模型的泛化性。</br>
&emsp;&emsp;这里提出GAP代替传统的FC层。主要思想是将每个分类对应最后一层MLPConv的输出特征图。对每个特征图取平均，后将得到的池化后的向量softmax得到分类概率。
```python
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
```
&emsp;&emsp;NiN的网络结构如上图所示，训练使用的优化器为Adam，学习率为0.001，在经过50次迭代后结果如下：
![alt text](<NIN 50 0.918.png>)
|      | NiN   | LeNet|VGG16|
| -------- | -------- | --------| --------|
|训练准确率  | 94.5% | 94.8%|100%|
| 测试准确率 | 91.8% |91.0%|93.3%|
| 训练时间/s | 1588.18 |1846.42|1409.13|

&emsp;&emsp;NiN使用全局平均池化来代替最后一个全连接层，能够有效地减少参数量，同时GAP层不用优化参数，可以避免过拟合。通过训练曲线可见NiN在实现了有效训练的同时其过拟合情况相比VGG也有明显改善。
##### 3.根据自身算力，尝试合适大小的GoogLeNet，效果如何？
&emsp;&emsp;GoogLeNet是2014年由Google团队提出的一种全新的深度学习结构，在这之前的AlexNet、VGG等结构都是通过增大网络的深度（层数）来获得更好的训练效果，但层数的增加会带来很多负作用，比如overfit、梯度消失、梯度爆炸等。inception的提出则从另一种角度来提升训练结果：能更高效的利用计算资源，在相同的计算量下能提取到更多的特征，从而提升训练结果。</br>
&emsp;&emsp;GoogLeNet网络的亮点有：

- 引入了Inception结构（融合不同尺度的特征信息）
- 使用1x1的卷积核进行降维以及映射处理
- 添加两个辅助分类器帮助训练
- 丢弃全连接层，使用平均池化层（大大减少模型参数）

&emsp;&emsp;Inception结构如下：
![alt text](image-4.png)
&emsp;&emsp;通过将Inception结构堆叠得到的GoogLeNet结构如图：
![alt text](image-5.png)

&emsp;&emsp;整个结构使用了9个inception modules线性堆叠而成，总共有22层（如果包含pooling层是27层），并且最后一个inception module后面增加了average pooling。</br>
&emsp;&emsp;这是一个相当深的结构，为了缓解梯度消失的问题，在这个结构的基础上增加两个辅助分类器。这两个分类器同样训练数据进行训练。并且，在损失函数部分，把这两部分的损失加到最后的损失函数中去。</br>
&emsp;&emsp;网络结构代码如下：(未定义和使用辅助分类器)
```python
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
```
&emsp;&emsp;在经过了50次迭代之后，训练结果如下：
![alt text](<googlenet 50 0.901.png>)
|      |  GoogLeNet  | LeNet|VGG16|
| -------- | -------- | --------| --------|
|训练准确率  | 99.5% | 94.8%|100%|
| 测试准确率 | 90.1% |91.0%|93.3%|
| 训练时间/s | 1673.24 |1846.42|1409.13|

&emsp;&emsp;相比于LeNet，GoogLeNet的测试准确率和训练准确率有所提升，可见GoogLeNet对于图像分类的有效性。但同时由训练曲线也可见GoogLeNet也存在一定的过拟合现象。而在设计上，GoogleNet通过模块化设计，避免了在参数数量上的线性增长，成为了参数量和计算量更高效的深层网络结构。但是由于我们简化了VGG的网络深度，所以在训练时间上这点并不明显。
##### 4.根据自身算力，尝试合适大小的ResNet
&emsp;&emsp;残差神经网络（Residual Neural Network，简称ResNet）属于深度学习模型的一种，其核心在于让网络的每一层不直接学习预期输出，而是学习与输入之间的残差关系。这种网络通过添加“跳跃连接”，即跳过某些网络层的连接来实现身份映射，再与网络层的输出相加合并。</br>
&emsp;&emsp;让我们聚焦于神经网络局部：如下图所示，假设我们的原始输入为x，而希望学出的理想映射为$f(x)$（作为下图上方激活函数的输入）。下图左图虚线框中的部分需要直接拟合出该映射$f(x)$，而右图虚线框中的部分则需要拟合出残差映射$f(x)-x$。残差映射在现实中往往更容易优化。实际中，当理想映射$f(x)$极接近于恒等映射时，残差映射也易于捕捉恒等映射的细微波动。下图是ResNet的基础架构–残差块（residual block）。在残差块中，输入可通过跨层数据线路更快地向前传播。
![alt text](image-6.png)

&emsp;&emsp;**具体的ResNet18结构为**
- 堆叠残差模块,每个残差模块有2个3x3conv layers
- 周期地,成倍增加卷积核的数量, 使用stride 2在空间上进行降采样(/2 in each
dimension)激活函数层输出大小周期性减半
- 在最开始增加额外的卷积层 (stem)
- 除了用于输出分类结果的FC 1000, 在最后没有额外的全连接层FC layer
- 卷积层后面是全局平均池化(Global average pooling layer)
- 理论上可以输入任何大小的图像

![alt text](image-7.png)

```python
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
```
&emsp;&emsp;使用以上网络结构在迭代50次之后结果如下：
![alt text](<RESNET 50 0.907.png>)
|      |  ResNet  | LeNet|VGG16|
| -------- | -------- | --------| --------|
|训练准确率  | 99.7% | 94.8%|100%|
| 测试准确率 | 90.7% |91.0%|93.3%|
| 训练时间/s | 1649.17 |1846.42|1409.13|

&emsp;&emsp;通过训练曲线可见ResNet实现了有效训练且训练效率较高。</br>
&emsp;&emsp;残差连接的引入让梯度能够更自由地在深层网络中流动，从而简化了优化过程。实验表明，ResNet比传统深层网络（如 VGG）更易于训练，能在训练中迅速达到较低的损失值。在传统的 CNN 中，随着层数增加，训练误差不仅没有降低，反而可能会增大，这被称为“退化问题”。而 ResNet 的残差连接能有效解决退化问题，因为即使某些层的特征不够有效，残差连接可以直接将之前层的特征传播到后层，确保模型不会因为层数增加而导致性能下降。
##### 5.探索下述论文中的可视化，尝试解释深度学习的工作原理
&emsp;&emsp;这篇论文《Visualizing and Understanding Convolutional Networks》由 Zeiler 和 Fergus 撰写，提出了一种新颖的可视化方法，来深入理解卷积神经网络（CNN）在分类任务中的内部工作机制。以下是论文的主要内容：
##### 引言：
&emsp;&emsp;论文指出，尽管大规模卷积网络在图像分类任务中取得了显著的性能，但人们对其内部运行机制和良好表现的原因缺乏清晰的理解。论文的目标是通过可视化手段揭示 CNN 的中间特征层及分类器的运作过程，帮助选择更优的网络架构，并提升分类性能。
##### 方法：
- 去卷积网络（Deconvnet）：论文使用去卷积网络作为主要的可视化工具。通过将 CNN 的中间特征层映射回像素空间，这种方法揭示了输入图像中哪些部分激活了某个特征图，从而提供直观的层级特征可视化。
- 反向投影过程：为了理解 CNN 的特征激活，作者将输入图像传入网络，记录某一层的激活特征，然后利用去卷积网络层逐层向下还原到输入图像的像素空间。该过程包括解池化、整流和过滤操作​。
##### 特征层可视化：
- 论文展示了 CNN 各层的特征可视化结果：低层特征响应于简单的边缘和角点，而高层特征则具备更高的类特异性（例如检测狗的脸部或鸟类的腿），表明 CNN 具有层级特征结构。这种逐层提取特征的机制，是 CNN 实现复杂模式识别的关键。
- 如下图显示了AlexNet在ImageNet数据集上特征可视化：
![alt text](6ca7c3d9eec4d5005b7ca432b896e701.png)
![alt text](c8f0b050b54d8684af519bfcc2e89675.png)
##### 训练过程中的特征演变：
- 通过观察特征在训练过程中的变化，作者发现底层特征收敛较快，而高层特征需要较长的训练时间才能形成。这表明在训练 CNN 时应让模型充分收敛，以便高层特征具备更强的识别能力。
##### 不变性分析：
- 论文还探讨了CNN对输入图像变换（如平移、旋转和缩放）的不变性。结果表明，低层特征对图像变化敏感，而高层特征表现出更大的不变性。网络的输出对平移和缩放具有稳定性，但对旋转的鲁棒性较低。
##### 模型架构选择：
- 可视化不仅用于解释已训练模型，还可以帮助选择合适的网络架构。通过观察第一和第二卷积层的特征，作者发现更小的卷积核和更小的步幅可以减少特征图中的混叠伪影，从而提高分类性能​。
##### 遮挡敏感性分析：
- 为验证模型是否真正聚焦于目标对象，作者采用遮挡实验，将图像的不同部分依次遮挡，并观察模型输出的变化。实验结果显示，当对象被遮挡时，模型的分类准确率显著下降，这证明 CNN 关注图像中的关键区域。
##### 跨数据集的特征泛化：
- 论文还展示了基于 ImageNet 训练的 CNN 在其他数据集（如 Caltech-101 和 Caltech-256）上的泛化能力，结果显著优于传统特征方法，表明 CNN 提取的特征具有较好的通用性。

&emsp;&emsp;这篇论文通过丰富的可视化分析，为理解 CNN 的内部结构提供了重要工具，并证明了这种理解有助于改进模型架构，从而提升分类效果。</br>
##### 对于特征层的可视化解释了深度学习的原理
&emsp;&emsp;论文中的可视化展示了 CNN 的层级结构：底层特征层通常会提取简单的边缘、颜色和纹理，而越往高层，提取的特征则越抽象、复杂。例如，在前几层中，特征图可能对直线、角点和基本形状做出响应，而在更深层次上，特征图则逐渐识别出特定物体的组成部分，甚至是完整的物体。这种特征提取过程反映了 CNN 的多层级特征学习机制，即从简单的低级特征到复杂的高级语义信息的逐步构建。</br>
&emsp;&emsp;可视化还显示了 CNN 对平移、缩放等几何变换的鲁棒性。低层特征通常对图像的平移、旋转等变化较为敏感，而高层特征更具有不变性和泛化性。例如，高层特征会忽略物体位置和姿态的变化，仍能正确识别出相似的结构。这种不变性有助于网络在分类时更稳定地识别同一类对象。</br>
&emsp;&emsp;论文的遮挡实验证明了模型对输入图像特定部分的依赖性。当遮挡物体关键区域时，分类概率显著下降，这表明模型学会了关注目标的特征区域。这种对重要区域的依赖说明 CNN 能够自动聚焦到图像中最有助于分类的部分，体现了CNN自动学习感知重要特征的能力。
