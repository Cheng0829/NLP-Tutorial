数据集下载: 链接：https://pan.baidu.com/s/17EL37CQ-FtOXhtdZHQDPgw 
提取码：0829

# 逻辑斯蒂回归

## 1.理论

### 1.1 多分类

若用logistc进行五分类，可以进行5次二分类，把情感标签当作5维向量。

softmax常用于多分类，当类别数为2时，和logistic等价。他把一些输入映射为0-1之间的实数，并且归一化保证和为1，因此多分类的概率之和也刚好为1。
而softmax是在输出层神经元输出结果$X(x_1,x_2···x_n)$之后，设$p_i=\frac{e^{x_i}}{\sum_{i=1}^ne^{x_i}}$

### 1.2 公式

$sigmoid(x)=\frac{1}{1+e^{-z}}$

$softmax(x)_i=\frac{e^{x_i}}{\sum_j e^{x_j}}$

$Cost:J(θ)=-\frac{1}{m}[{\sum_{i=1}^m}((y^{(i)})logh_θ(x^{(i)})+(1-y^{(i)}))log(1-h_θ(x^{(i)}))+λ{\sum_{j=1}^nθ_j^2}]$

$hypothesis:h_θ(x)=sigmoid(θ^Tx)=\frac{1}{1+e^{-θ^Tx}}$

$gradient=\frac{\partial J(θ_0,θ_1)}{\partial θ_j}=\frac{1}{m}{\sum_{i=1}^m}[(h_θ(x^{(i)})-y^{(i)})x_j^{(i)}+λθ_j]$

$GradientDescent:θ_j=θ_j-α\frac{\partial J(θ_0,θ_1)}{\partial θ_j}=θ_j-\frac{α}{m}{\sum_{i=1}^m}[(h_θ(x^{(i)})-y^{(i)})x_j^{(i)}+λθ_j]$

![ ](img/logistic.png)

## 2.实验

### 2.1 实验步骤

1) **读取数据**
2) **训练**
   1) 构建词频矩阵,进行标准化
   2) 构建逻辑回归模型,**自动拟合训练**
3) **预测**

