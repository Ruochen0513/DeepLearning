## <center>多层感知机实验报告
##### <center>2213530 张禹豪 智能科学与技术
#### 一、实验目的
- 了解最简单的深度网络多层感知机(MLP)的网络模型
- 从零构建一个简易的多层感知机并使用MNIST数据集进行训练
- 计算多层感知机网络的参数量，实际感受深度学习网络参数量的庞大
- 更改多层感知机的隐藏层数和每一层神经元数量以改变网络的参数量，观察网络参数量的改变对训练误差、测试误差即网络性能的影响
- 尝试不同的激活函数和损失函数，观察其对网络性能的影响
- 了解模型复杂度（隐藏层层数、隐藏层神经元数）与训练误差和测试误差的关系
- 观察模型结果的过拟合现象并使用多种方法减轻过拟合问题

#### 二、实验原理
##### 多层感知机<br />
&emsp;&emsp;多层感知机(MLP)也叫人工神经网络(ANN)，通过在网络中加入一个或多个隐藏层来克服线性模型的限制，使其能处理更普遍的函数关系类型。要做到这一点，最简单的方法是将许多全连接层堆叠在一起。 每一层都输出到上面的层，直到生成最后的输出。我们可以把前层看作表示，把最后一层看作线性预测器。其具体结构如下图所示：
![alt text](MLP.jpg)<br />
&emsp;&emsp;对于一个简易的单层线性打分函数，即$$f = Wx\quad\quad x∈R^D,W∈R^{C×D}$$
&emsp;&emsp;而多层感知机就是将上述单层函数全连接起来，例如2层神经网络，即$$f = W_2max(0,W_1x) \quad\quad x∈R^D,W_1∈R^{H×D},W_2∈R^{C×H}$$
&emsp;&emsp;在上述函数中我们加入了一个max()函数，它被称为激活函数，如果不使用激活函数，上面的隐藏单元由输入的仿射函数给出， 而输出只是隐藏单元的仿射函数。仿射函数的仿射函数本身就是仿射函数，但是我们之前的线性模型已经能够表示任何仿射函数。所以如果不使用激活函数，我们的神经网络只能去解决线性问题，只有添加激活函数，才能为线性分类器增添非线性特性，从而使得神经网络能够拟合非线性函数。<br />
##### 激活函数
&emsp;&emsp;所谓激活函数（Activation Function），就是在人工神经网络的神经元上运行的函数，负责将神经元的输入映射到输出端。常见的激活函数如下所示：
- ReLu函数<br />
&emsp;&emsp;ReLU，全称为：Rectified Linear Unit，是一种人工神经网络中常用的激活函数，通常意义下，其指代数学中的斜坡函数，即
$$ f(x)=max(0,x)$$
![alt text](relu.jpg)
- Sigmoid函数<br />
&emsp;&emsp;Sigmoid函数是一个在生物学中常见的S型函数，也称为S型生长曲线。在深度学习中，由于其单增以及反函数单增等性质，Sigmoid函数常被用作神经网络的激活函数，将变量映射到[0,1]之间。
$$S(x) = \frac{1}{1+e^{-x}} $$
![alt text](sigmoid.jpg)<br />

##### 损失函数
&emsp;&emsp;通过上述多层网络和激活函数我们就建立了一个多层感知机，为了判断一个网络模型是否有效，我们需要有一个损失函数去衡量模型预测值与真实值之间的差异，优化网络模型的过程就是最小化损失函数的过程。常见的损失函数如下：
- 交叉熵损失（Cross-Entropy Loss，CE）
$$CE=-\frac{1}{n}\sum_{i=1}^{n}\sum_{j=1}^{C}y_{ij}log(\hat{y}_{ij})$$
- 均方误差损失（Mean Squared Error, MSE）
$$MSE=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2$$

##### 优化算法
&emsp;&emsp;正如上述所说，优化网络模型的过程就是最小化损失函数的过程。优化算法在深度学习中起着至关重要的作用，它们通过调整模型的参数来最小化损失函数。不同的优化算法有不同的特点和适用场景。以下是一些常见的优化算法：
- 随机梯度下降(Stochastic Gradient Descent, SGD)<br />

&emsp;&emsp;SGD 是最基础的优化算法之一。它通过随机选择一样本（或一个小批量样本）来估计梯度，并据此更新参数。
- 动量 (Momentum)<br />

&emsp;&emsp;动量方法通过引入一个“动量”项来加速SGD的收敛。动量项累积了过去的梯度，有助于模型在平坦区域更快地移动
- Adam (Adaptive Moment Estimation)<br />

&emsp;&emsp;Adam结合了动量的优点，通过维护梯度的一阶矩估计和二阶矩估计来动态调整学习率。

##### 反向传播
&emsp;&emsp;反向传播（Backpropagation）是神经网络训练中的一项关键技术，用于高效地计算损失函数对网络中每个参数的梯度。通过这些梯度，优化算法可以更新网络参数以最小化损失函数。反向传播基于链式法则和动态规划的思想，能够在大规模神经网络中有效地进行梯度计算。
#### 三、基础网络模型建立
##### 定义所使用到的激活函数和损失函数
```python
# 定义激活函数
def relu(X):
    return torch.max(input=X, other=torch.zeros_like(X))
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition
def sigmoid(X):
    return 1 / (1 + torch.exp(-X))
# 定义交叉熵损失函数
def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])
# 定义均方误差损失函数
def mse_loss(y_hat, y):
    return ((y_hat - y.to(y_hat.dtype)) ** 2).mean()
```
&emsp;&emsp;根据上述原理中的各函数定义，我们可以很容易定义出激活和损失函数
##### MLP网络模型建立
```python
# 定义模型类
class MLP(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1,
     num_hiddens2):
        super(MLP, self).__init__()
        # 初始化模型参数
        self.W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens1, 
        requires_grad=True) * 0.01)
        self.b1 = nn.Parameter(torch.zeros(num_hiddens1, 
        requires_grad=True))
        self.W2 = nn.Parameter(torch.randn(num_hiddens1, num_hiddens2,
         requires_grad=True) * 0.01)
        self.b2 = nn.Parameter(torch.zeros(num_hiddens2,
         requires_grad=True))
        self.W3 = nn.Parameter(torch.randn(num_hiddens2, num_outputs,
         requires_grad=True) * 0.01)
        self.b3 = nn.Parameter(torch.zeros(num_outputs,
         requires_grad=True))
        self.num_inputs = num_inputs
    def forward(self, X):
        X = X.reshape((-1, self.num_inputs))
        H = relu(X @ self.W1 + self.b1)
        C = relu(H @ self.W2 + self.b2)
        return softmax(C @ self.W3 + self.b3)
```
&emsp;&emsp;在这里我们定义了一个三层（包括两层隐藏层和一层输出层）感知机。我们使用nn.Parameter()函数来定义了各层网络的参数，包括每层的权重和偏置。在这三层网络中，我对两层隐藏层都使用了relu激活函数，而输出层使用了softmax激活函数，softmax可以将最终输出层的结果激活为概率分布，便于我们进行分类。
##### 定义训练函数
```python
# 计算准确率
def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()
# 定义训练函数
def train(net, train_iter, test_iter, loss, num_epochs, updater, device=None):
    train_losses, test_accs, train_accs = [], [], []
    # 训练模式
    for epoch in range(num_epochs):
        net.train()
        train_loss, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            updater.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y).mean()
            l.backward()
            updater.step()

            train_loss += l.item() * y.shape[0]
            train_acc_sum += accuracy(y_hat, y) * y.shape[0]
            n += y.shape[0]

        train_losses.append(train_loss / n)
        train_accs.append(train_acc_sum / n)

        # 测试模式
        net.eval()
        test_acc_sum, m = 0.0, 0
        with torch.no_grad():
            for X, y in test_iter:
                X, y = X.to(device), y.to(device)
                y_hat = net(X)
                test_acc_sum += accuracy(y_hat, y) * y.shape[0]
                m += y.shape[0]
        test_accs.append(test_acc_sum / m)

        print(f'Epoch {epoch + 1}, Loss: {train_losses[-1]:.3f}, 
        Train Acc: {train_accs[-1]:.3f}, Test Acc: {test_accs[-1]:.3f}')
    return train_losses, test_accs, train_accs
```
&emsp;&emsp;accuracy()函数用于计算模型在给定的数据集上的准确率，即模型正确预测的比例。<br/>
&emsp;&emsp;train()函数负责执行模型的训练过程。它接受模型、训练数据加载器、测试数据加载器、损失函数、训练轮数、优化器以及可选的设备参数。该函数会在每个epoch中迭代训练数据集，计算损失，反向传播更新模型参数，并定期评估模型在测试数据集上的表现。
##### 绘制训练指标
```python
# 画出损失函数和准确率随 epoch 变化的曲线
def plot_metrics(train_losses, test_accs, train_accs, num_epochs):
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, 'r-', label='Training Loss')
    plt.plot(epochs, test_accs, 'b-', label='Test Accuracy')
    plt.plot(epochs, train_accs, 'g-', label='Train Accuracy')
    plt.xlabel('Epoch')
    plt.xlim([1, num_epochs])
    plt.ylim([0, 1])
    plt.ylabel('Accuracy / Loss')
    plt.title('Training Loss, Test Accuracy, and Train Accuracy vs.
     Epoch')
    plt.legend()
    plt.show()
```
&emsp;&emsp;plot_metrics()函数用于绘制训练过程中损失值和准确率的变化曲线，帮助我们直观地了解模型的学习情况。
##### 主函数
```python
def main():
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else 
    "cpu")
    print(f"Using device: {device}")
    # 数据预处理
    transform = transforms.Compose([transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,))])
    # 加载MNIST数据集
    train_dataset = datasets.MNIST(root='./mnist_data', train=True,
    download=False, transform=transform)
    test_dataset = datasets.MNIST(root='./mnist_data', train=False, 
    download=False, transform=transform)
    train_iter = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_iter = DataLoader(test_dataset, batch_size=256, shuffle=False)
    # 定义参数
    num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 
    256, 256
    # 实例化模型
    net = MLP(num_inputs, num_outputs, num_hiddens1, num_hiddens2).to(device)
    # 训练模型
    num_epochs, lr = 10, 0.1
    updater = torch.optim.SGD(net.parameters(), lr=lr)
    train_losses, test_accs, train_accs = train(net, train_iter, 
    test_iter, cross_entropy, num_epochs, updater,device=device)
    plot_metrics(train_losses, test_accs, train_accs, num_epochs)
```
&emsp;&emsp;main函数是程序的入口点，它完成了以下工作：
- 检查是否有可用的GPU设备，如果有的话就使用GPU，否则使用CPU。
- 对MNIST数据集进行预处理，包括转换为张量格式和标准化。
- 加载训练集和测试集，并创建相应的数据加载器。
- 设置模型参数，实例化模型，并将其移动到指定的设备上。
- 定义训练轮数和学习率，并选择优化算法（这里是SGD）。
- 调用train函数开始训练过程，并记录训练和测试的损失及准确率。
- 使用plot_metrics函数绘制训练过程中的性能指标。
#### 四、实验要求及结果展示
##### 1、画出训练过程中训练损失函数、训练误差、测试误差随迭代epoch的变化曲线
&emsp;&emsp;在如下条件下：三层感知机(两层隐藏层是ReLu激活函数、输出层是Softmax激活函数)，损失函数是交叉熵损失函数，优化算法是随机梯度下降，所使用数据集为MNIST数据集。训练过程性能曲线如下图：
![alt text](MLP1.png)
&emsp;&emsp;可见在经过10次迭代之后测试集准确率达到95.96%，网络效果较好。
##### 2、计算网络的参数量（权重数+偏置数）
- 输入层到第一个隐藏层：<br/>
&emsp;&emsp;权重矩阵$W_1$的参数量：784×256=200,704<br/>
&emsp;&emsp;偏置向量$b_1$的参数量：256<br/>
&emsp;&emsp;总参数量：200,704+256=200,960
- 第一个隐藏层到第二个隐藏层：<br/>
&emsp;&emsp;权重矩阵$W_2$的参数量：256×256=65,536<br/>
&emsp;&emsp;偏置向量$b_2$的参数量：256<br/>
&emsp;&emsp;总参数量：65,536+256=65,792
- 第二个隐藏层到输出层：<br/>
&emsp;&emsp;权重矩阵$W_3$的参数量：256×10=2,560<br/>
&emsp;&emsp;偏置向量$b_3$的参数量：10<br/>
&emsp;&emsp;总参数量：2,560+10=2,570
- 整个网络总参数量：200,960+65,792+2,570=**269,322**
##### 3、更改隐藏层数或者每一隐藏层神经元数量，观察要求1中3个变化曲线及要求2中的参数量，总结参数量对性能的影响
- 就初始条件，将两个隐藏层的神经元数量从256变为32，此时参数量为26506，实验结果如下图:
![alt text](mlp3.png)
&emsp;&emsp;在经过十次迭代之后测试集准确率为：91.6%
- 就初始条件，将两个隐藏层改为一个隐藏层，此时参数量为203530，实验结果如下图:
![alt text](mlp4.png)
&emsp;&emsp;在经过十次迭代之后测试集准确率为：95.5%<br/>
&emsp;&emsp;通过比较初始条件和以上两种情况下不同的性能曲线，我们可以发现：参数量越大，训练过程中的损失函数会越小，训练误差会越小，测试误差会越小。<br/>
&emsp;&emsp;而由于过拟合的影响，当参数量非常大时，测试误差会随着迭代次数的增长而先减后增。<br/>
&emsp;&emsp;综上所述，参数量对性能的影响是多方面的，更多的参数意味着模型具有更强的表达能力，可以拟合更复杂的函数和模式，同时拟合的更加好。过多的参数使得模型容易过拟合训练数据，即模型在训练数据上表现很好，但在未见过的测试数据上表现较差。
##### 4、尝试不同的激活函数，观察要求1中3个变化曲线
```python
def sigmoid(X):
    return 1 / (1 + torch.exp(-X))
```
&emsp;&emsp;我们将两层隐藏层的激活函数从ReLu改为上面的sigmoid激活函数，结果如下：
![alt text](mlp5.png)
&emsp;&emsp;与初始条件比较可见，就本网络模型和数据集，ReLu激活函数明显优于sigmoid激活函数。
##### 5、尝试不同的损失函数，观察要求1中3个变化曲线
&emsp;&emsp;我将交叉熵函数改为使用MSE损失函数，注意此时输出层不再使用softmax激活函数，代码添加MSE损失函数定义：
```python
# 均方误差损失函数
class MSELoss(nn.Module):
    def __init__(self,num_outputs):
        super(MSELoss, self).__init__()
        self.num_outputs = num_outputs
    def forward(self, output, target, params=None):
        target_one_hot = torch.nn.functional.one_hot(target,
         num_classes=self.num_outputs).float()
        mse = torch.square(output - target_one_hot)
        return mse.mean()
    ......其他不变......
def main():   
    # 训练模型
    num_epochs, lr = 20, 0.1
    updater = torch.optim.SGD(net.parameters(), lr=lr)
    loss = MSELoss(num_outputs)
    train_losses, test_accs, train_accs = train(net, train_iter, 
    test_iter, loss , num_epochs, updater,device=device)
```
&emsp;&emsp;上述训练结果如下：
![alt text](mlp15.jpg)
&emsp;&emsp;在经过20次迭代之后测试准确度为93.2%，低于使用交叉熵损失函数的训练结果，所以对于多分类问题使用交叉熵损失函数优于使用均方误差损失函数。
#### 五、附加题
##### 1、解释模型复杂度（隐藏层层数、隐藏层神经元数）与训练误差和测试误差的关系
&emsp;&emsp;模型复杂度通常由以下几个因素决定：
- 隐藏层层数：隐藏层的数量越多，模型的能力越强，能够捕捉到更复杂的模式。
- 隐藏层神经元数：每个隐藏层中的神经元数量越多，模型的表示能力越强。<br/>

**训练误差**：训练误差是指在训练数据集上模型的性能。以下是如何随着模型复杂度的增加，训练误差的变化趋势：
- 低复杂度：如果模型太简单（隐藏层层数和神经元数都较少），它可能无法捕捉到数据中的所有重要模式，导致较高的训练误差。
- 适中复杂度：随着模型复杂度的增加，模型能够更好地拟合训练数据，训练误差通常会下降。
- 高复杂度：如果模型过于复杂（隐藏层层数和神经元数过多），它可能会过拟合训练数据，即模型不仅捕捉到了数据中的模式，还捕捉到了噪声和细节，这时训练误差会非常低，甚至接近于零。<br/>

**测试误差**：测试误差是指在未见过的新数据集上模型的性能。以下是如何随着模型复杂度的增加，测试误差的变化趋势：
- 低复杂度：模型可能因为缺乏足够的表示能力而无法泛化到新数据上，导致较高的测试误差。
- 适中复杂度：在某个点，模型复杂度达到最佳，能够在训练集上学习到有用的模式，并且能够很好地泛化到测试集上，这时测试误差最低。
- 高复杂度：模型过拟合训练数据，对新数据的泛化能力变差，测试误差会上升。
##### 2、观察训练误差和测试误差的走势，观察是否有过拟合现象；如果观察不出，尝试更换更大的数据集或者更换网络结构，使得过拟合现象出现
&emsp;&emsp;通过在上述初始网络模型下训练MNIST数据集，在迭代20次之后的性能曲线如下图所示：
![alt text](mlp6.png)
&emsp;&emsp;如图可见在这种条件下，训练准确率保持较高的水平，而测试准确率始终比训练准确率低一点，不能很好地吻合，而且测试准确率在迭代中还出现突然下降的情况，这些情况说明该网络训练过程中出现了过拟合的现象，该模型过拟合训练数据，对新数据的泛化能力变差。
##### 3、在损失函数中增加$l_2$正则化，观察训练误差和测试误差
&emsp;&emsp;要在损失函数中增加$l_2$正则化，我们需要对网络的权重应用正则化项，并在计算损失时将其添加到原始损失上。$l_2$正则化项是权重向量的每个元素的平方和，乘以一个正则化系数。<br/>
&emsp;&emsp;以下是修改后的train函数：
```python
# 定义训练函数
def train(net, train_iter, test_iter, loss, num_epochs, updater, 
device=None, weight_decay=0.01):
    train_losses, test_accs, train_accs = [], [], []

    # 训练模式
    for epoch in range(num_epochs):
        net.train()
        train_loss, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            updater.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y).mean()
            
            # 添加L2正则化
            l2_penalty = 0
            for param in net.parameters():
                l2_penalty += torch.norm(param)**2
            l += weight_decay * l2_penalty

            l.backward()
            updater.step()

            train_loss += l.item() * y.shape[0]
            train_acc_sum += accuracy(y_hat, y) * y.shape[0]
            n += y.shape[0]

        train_losses.append(train_loss / n)
        train_accs.append(train_acc_sum / n)

      ......以下部分不变......
```
&emsp;&emsp;在这段代码中，我们定义了一个 weight_decay 参数，它是$l_2$正则化的系数。在每次迭代中，我们计算网络所有参数的平方和，然后乘以 weight_decay 并将其添加到损失中。然后，我们像以前一样执行反向传播和参数更新。<br/>
&emsp;&emsp;增加$l_2$正则化通常会减少过拟合，因为正则化项惩罚了权重的大小，鼓励模型学习更简单的映射。这可能会导致训练误差略有增加，但通常能提高模型在测试数据上的泛化能力，从而降低测试误差。<br/>
&emsp;&emsp;通过观察下面训练误差和测试误差随epoch的变化，我们可以看到$l_2$正则化的效果:
![alt text](mlp7.png)
&emsp;&emsp;正如我们所料，在使用$l_2$正则化之后确实会导致训练误差略有增加，但同时提高了模型在测试数据上的泛化能力。
##### 4、使用dropout，观察其对泛化的影响
**dropout**<br/>
&emsp;&emsp;Dropout 是一种非常有效的正则化技术，通过在训练过程中随机丢弃一部分神经元（及其连接）来减少过拟合。具体来说，Dropout 在每次前向传播时，随机选择一部分神经元，并将它们的输出设置为零。这些被丢弃的神经元在当前前向传播和反向传播中不会参与任何计算。为了观察dropout对于泛化的影响，我们在网络模型中应用dropout：
```python
# 定义模型类
class MLP(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2, dropout_prob=0.5):
        super(MLP, self).__init__()
        # 初始化模型参数
        self.W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens1, requires_grad=True) * 0.01)
        self.b1 = nn.Parameter(torch.zeros(num_hiddens1, requires_grad=True))
        self.W2 = nn.Parameter(torch.randn(num_hiddens1, num_hiddens2, requires_grad=True) * 0.01)
        self.b2 = nn.Parameter(torch.zeros(num_hiddens2, requires_grad=True))
        self.W3 = nn.Parameter(torch.randn(num_hiddens2, num_outputs, requires_grad=True) * 0.01)
        self.b3 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
        self.num_inputs = num_inputs
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, X):
        X = X.reshape((-1, self.num_inputs))
        H = relu(X @ self.W1 + self.b1)
        H = self.dropout(H)  # 应用Dropout
        C = relu(H @ self.W2 + self.b2)
        C = self.dropout(C)  # 应用Dropout
        return softmax(C @ self.W3 + self.b3)
```
- 在MLP类的构造函数中添加了一个dropout_prob参数，用于控制Dropout的比例。
- 在前向传播过程中，在每个隐藏层的输出后应用Dropout。<br/>

&emsp;&emsp;添加了dropout的网络训练结果如下：
![alt text](mlp-8.png)
&emsp;&emsp;通过比较初始模型数据和上图可以发现：
- 训练损失：Dropout导致训练损失略有增加，因为部分神经元被随机丢弃，模型的学习能力受到一定程度的限制。
- 训练准确率：训练准确率略有下降，因为模型在训练过程中部分神经元被丢弃，减少了过拟合的风险。
- 测试准确率：测试准确率有所提高，因为Dropout有助于减少模型的过拟合，提高模型的泛化能力。
##### 5、使用batch normalization，观察其对训练误差和测试误差的影响
**batch normalization**<br/>
&emsp;&emsp;批归一化（Batch Normalization, BN）是一种广泛应用于深度学习模型中的技术，旨在通过规范化前一层的输出来改善神经网络的训练过程。其作用如下：
- 加速训练：通过减少内部协变量偏移，批归一化可以允许使用更高的学习率，而不会遇到梯度消失或梯度爆炸的问题，从而加速模型的收敛速度。
- 正则化效果：批归一化具有一定的正则化效果，因为它引入了一定程度的噪声（尤其是在小批量的情况下），这有助于防止过拟合。
- 减少对初始化的依赖：批归一化可以减少模型对权重初始化的敏感性，因为规范化过程有助于保持激活值在一个合理的范围内，即使初始权重选择不当，也能较快地进入有效的训练状态。
- 简化超参数调整：批归一化可以减少对其他形式的正则化（如Dropout）的需求，有时甚至可以减少对学习率衰减策略的依赖，因为批归一化本身就有助于稳定训练过程。
```python
# 定义模型类
class MLP(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_inputs, num_hiddens1),
            nn.BatchNorm1d(num_hiddens1),  # 添加批归一化层
            nn.ReLU(),
            nn.Linear(num_hiddens1, num_hiddens2),
            nn.BatchNorm1d(num_hiddens2),  # 添加批归一化层
            nn.ReLU(),
            nn.Linear(num_hiddens2, num_outputs)
        )
    def forward(self, X):
        return self.net(X)
```
&emsp;&emsp;在这里，我们使用了PyTorch的nn.Sequential来构建模型，这使得我们可以更方便地插入nn.BatchNorm1d层。批归一化层被放置在每个线性变换和ReLU激活函数之间。这样做可以帮助标准化每一层的输入，从而可能加快训练速度并改善模型性能。训练结果如下：
![alt text](mlp9.png)
&emsp;&emsp;观察以上结果可以发现。使用了Batch Normalization之后，使得模型对初始权重的选择不那么敏感，模型能更快地进入有效的训练状态，训练、测试误差很快降到了非常低；<br/>
&emsp;&emsp;批归一化在训练过程中引入了一定程度的噪声，这有助于防止过拟合，改善了模型的泛化能力，降低了测试误差；<br/>
&emsp;&emsp;批量归一化通过规范化每个小批量数据的输入，减少了所谓的“内部协变量偏移”，这使得网络更容易学习，有效降低了训练误差，在本次训练中，训练误差很快降为了零。
##### 6、训练一个神经网络，解决亦或问题
&emsp;&emsp;针对此问题，我重新训练了一个双层感知机，其中包括一层隐藏层和一层输出层，隐藏层使用ReLu激活函数，输出层不使用激活函数，损失函数为MSE，优化算法为随机梯度下降。具体代码如下：
```python
import torch
from torch import nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
# 定义模型类
class MLP(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_inputs, num_hiddens1),
            nn.ReLU(),
            nn.Linear(num_hiddens1, num_outputs),
        )

    def forward(self, X):
        return self.net(X)

# 计算准确率
def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()

# 定义训练函数
def train(net, train_iter, test_iter, loss, num_epochs, updater, device=None):
    train_losses, test_accs, train_accs = [], [], []

    # 训练模式
    for epoch in range(num_epochs):
        net.train()
        train_loss, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            updater.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y).mean()
            l.backward()
            updater.step()

            train_loss += l.item() * y.shape[0]
            n += y.shape[0]
        train_losses.append(train_loss / n)
        # 测试模式
        net.eval()
        test_acc_sum, m = 0.0, 0
        with torch.no_grad():
            for X, y in test_iter:
                X, y = X.to(device), y.to(device)
                y_hat = net(X)
                test_acc_sum += accuracy(y_hat, y) * y.shape[0]
                m += y.shape[0]
        test_accs.append(test_acc_sum / m)

        print(
            f'Epoch {epoch + 1}, Loss: {train_losses[-1]:.3f},  Test Acc: {test_accs[-1]:.3f}')

    return train_losses, test_accs


# 画出损失函数和准确率随 epoch 变化的曲线
def plot_metrics(train_losses, test_accs, num_epochs):
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, 'r-', label='Training Loss')
    plt.plot(epochs, test_accs, 'b-', label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.xlim([1, num_epochs])
    plt.ylim([0, 1])
    plt.ylabel('Accuracy / Loss')
    plt.title('Training Loss, Test Accuracy vs. Epoch')
    plt.legend()
    plt.show()
def get_input():
    x1 = int(input("请输入第一个数字（0或1）："))
    x2 = int(input("请输入第二个数字（0或1）："))
    input_tensor = torch.tensor([[x1, x2]], dtype=torch.float32)
    return input_tensor

def output_result(net, input_tensor):
    y_hat = net(input_tensor)
    rounded_prediction = torch.round(y_hat).int().item()
    print(f"预测结果为：{rounded_prediction}")

def main():
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 训练数据
    X_train = torch.tensor([[0, 0], [0, 1], [1, 0],[1, 1]], dtype=torch.float32)
    y_train = torch.tensor([[0], [1], [1],[0]], dtype=torch.float32)
    # 测试数据
    X_test = torch.tensor([[1, 1]], dtype=torch.float32)
    y_test = torch.tensor([[0]], dtype=torch.float32)

    train_iter = DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=1, shuffle=True)
    test_iter = DataLoader(torch.utils.data.TensorDataset(X_test, y_test), batch_size=1, shuffle=False)
    # 定义参数
    num_inputs, num_outputs, num_hiddens1 = 2, 1, 8
    # 实例化模型
    net = MLP(num_inputs, num_outputs, num_hiddens1).to(device)
    # 训练模型
    num_epochs, lr = 100, 0.1
    updater = torch.optim.SGD(net.parameters(), lr=lr)
    criterion = nn.MSELoss()  # 均方误差损失函数

    train_losses, test_accs = train(net, train_iter, test_iter,criterion , num_epochs, updater,
                                                device=device)
    plot_metrics(train_losses, test_accs, num_epochs)

    input_tensor = get_input()  # 获取用户输入
    output_result(net, input_tensor)  # 输出预测结果
if __name__ == '__main__':
    main()
```
&emsp;&emsp;通过以上网络训练异或问题数据集得到结果如下：
![alt text](mlp10.jpg)
&emsp;&emsp;可见测试准确率很快达到100%，之后再次实际测试一下结果如下：
![alt text](mlp11.jpg)![alt text](mlp12.jpg)![alt text](mlp13.jpg)![alt text](mlp14.jpg)<br/>
&emsp;&emsp;可见我们的网络是非常有效的。