# CNN

卷积神经网络一般由**卷积层、池化层**(汇聚层)和**全连接层**构成．

**整个流程:**
![ ](img/CNN-%E6%95%B4%E4%BD%93%E6%B5%81%E7%A8%8B.jpg)

## 1.基本卷积运算(互相关)

![ ](img/CNN-%E5%8D%B7%E7%A7%AF%E5%B1%82.png)

```python
import torch
from torch import nn
def corr2d(X, K):  # 二维卷积(互相关)运算
    h, w = K.shape
    Y = torch.zeros((X.shape[0]-h+1, X.shape[1]-w+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = (X[i:i+h, j:j+w] * K).sum()
    return Y
X = torch.tensor([[3,0,1,2,7,4],[1,5,8,9,3,1],[2,7,2,5,1,3],[0,1,3,1,7,8],[4,2,1,6,2,8],[2,4,5,2,3,9]])
K = torch.tensor([[1,0,-1],[1,0,-1],[1,0,-1]])
# K为权值矩阵
Y = corr2d(X, K) # Y 为已知的输出结果 
print(Y)
```

## 2.卷积层

![ ](img/CNN-%E8%BF%90%E7%AE%97.png)
神经元和过滤器的通道数必须一致,同时过滤器个数即为下一层神经元的通道数

```python
# 计算卷积层
import torch
from torch import nn
X = torch.tensor([[3,0,1,2,7,4],[1,5,8,9,3,1],[2,7,2,5,1,3],[0,1,3,1,7,8],[4,2,1,6,2,8],[2,4,5,2,3,9]],dtype=torch.float)
conv2d = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=(3,3),padding=(0,0),stride=(1,1))# 注意这⾥是两侧分别填充1⾏或列，所以在两侧⼀共填充2⾏或列
X = X.view((1,1) + X.shape) #增加两个维度:# (1, 1)代表批量⼤⼩和通道数均为1
Y = conv2d(X)
# 权值矩阵隐藏在conv2d对象内部,具体操作可见梯度下降部分
print(conv2d)
print(Y.view(Y.shape[2:]))# 降维:排除不关⼼的前两维：批量和通道
```

```python
# 多通道卷积运算
#对每一条个通道做互相关运算,然后累加
import torch
def corr2d(X, K): # 基本卷积运算
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+h, j:j+w] * K).sum()
    return Y
    
K = torch.tensor([[[1,0,-1],[1,0,-1],[1,0,-1]]]) 
X = torch.tensor([[[3,0,1,2,7,4],[1,5,8,9,3,1],[2,7,2,5,1,3],[0,1,3,1,7,8],[4,2,1,6,2,8],[2,4,5,2,3,9]]])
def corr2d_multi_in(X, K):
    # X和K是二维的,则应该多加一层括号,变成三维(表示有多个通道)
    # 沿着X和K的第0维（通道维）分别计算再相加
    res = corr2d(X[0,:,:], K[0,:,:])
    #print(res)
    for i in range(1, X.shape[0]): #X.shape[0]是通道数,对每一条个通道做互相关运算,然后累加
        res += corr2d(X[0,:,:], K[0,:,:])
    return res

a = corr2d_multi_in(X, K)
print(a)

def corr2d_multi_in_out(X, K):
    # 对K的第0维遍历，每次同输⼊X做互相关计算。所有结果使⽤stack函数合并在⼀起
    return torch.stack([corr2d_multi_in(X, k) for k in K])

K = torch.stack([K, K + 1, K + 2]) # 三个通道
print('K:',K)
a = corr2d_multi_in_out(X, K)
print(a)
```

## 3.池化层

池化层的⼀个主要作⽤是缓解卷积层对位置的过度敏感性
![ ](img/CNN-%E6%B1%A0%E5%8C%96%E5%B1%82.png)

```python
# 最大池化
X = [[1,3,2,1,3],[2,9,1,1,5],[1,3,2,3,2],[8,3,5,1,0],[5,6,1,2,9]]
# X = torch.arange(16, dtype=torch.float).view((1, 1, 4, 4)) # 创建一个4x4的等差矩阵,通道批量均为1
X = torch.tensor(X, dtype=torch.float).view((1,1,5,5))
print(X)
pool2d = nn.MaxPool2d(3,padding=0, stride=1)
print(pool2d(X))
```

## 4.全连接层

1x1的卷积层被当作保持⾼和宽维度形状不变的全连接层使⽤.我们因此可以通过调整网络层之间的通道数来控制模型复杂度
![ ](img/CNN-%E5%85%A8%E8%BF%9E%E6%8E%A5.png)

```python
# 全连接
import torch
def corr2d_multi_in_out_1x1(X, K):
    # 计算前需要对数据的形状进行调整(同时XK)
    # 做1x1卷积时，以上函数与之前实现的互相关运算函数corr2d_multi_in_out等价
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.view(c_i, h * w) # 计算前需要对数据的形状进行调整
    K = K.view(c_o, c_i) # 计算前需要对数据的形状进行调整
    Y = torch.mm(K, X) # 全连接层的矩阵乘法
    return Y.view(c_o, h, w)
```

## 5.梯度下降

```python
# 梯度下降
import torch
from torch import nn ,optim
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))
    def forward(self, x):
        return corr2d(x, self.weight) + self.bias

def corr2d(X, K): 
    h, w = K.shape
    Y = torch.zeros((X.shape[0]-h+1, X.shape[1]-w+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = (X[i:i+h, j:j+w] * K).sum()
    return Y

X = torch.tensor([[1,0,1,1,0,0],[1,0,1,0,0,1],[0,0,0,0,1,1],[0,1,0,1,0,0],[0,0,1,0,0,0],[1,1,1,0,0,0]],dtype=torch.float)
# X不能大于1,否则应该归一化
K = torch.tensor([[1,0,-1],[1,0,-1],[1,0,-1]])

Y = corr2d(X, K)
conv2d = Conv2D(kernel_size=K.shape) 
step = 20
lr = 0.01 # 学习率
Y_hat = conv2d(X)
for i in range(step):
    Y_hat = conv2d(X)
    e = ((Y_hat-Y)**2).sum()
    e.backward()
    # 梯度下降
    conv2d.weight.data -= lr * conv2d.weight.grad
    conv2d.bias.data -= lr * conv2d.bias.grad
    # 梯度清0
    conv2d.weight.grad.fill_(0)
    conv2d.bias.grad.fill_(0)
    print('Step %d, loss %.3f' % (i + 1, e.item()))
```
