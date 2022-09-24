"""
Task: 生成单词的特征嵌入矩阵,并进行图形化显示
Author: ChengJunkai @github.com/Cheng0829
Email: chengjunkai829@gmail.com
Date: 2022/09/05
Reference: Tae Hwan Jung(Jeff Jung) @graykode
"""

import numpy as np
import torch, os, sys, time
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

'''1.数据预处理'''
def pre_process(sentences):
    # 分词
    word_sequence = " ".join(sentences).split() # ['apple', 'banana', 'fruit', 'banana', 'orange', 'fruit', 'orange', 'banana', 'fruit', 'dog', 'cat', 'animal', 'cat', 'monkey', 'animal', 'monkey', 'dog', 'animal']
    # 去重
    word_list = []
    '''
    如果用list(set(word_sequence))来去重,得到的将是一个随机顺序的列表(因为set无序),
    这样得到的字典不同,保存的上一次训练的模型很有可能在这一次不能用
    (比如上一次的模型预测碰见i:0,love:1,就输出you:2,但这次模型you在字典3号位置,也就无法输出正确结果)
    '''
    for word in word_sequence:
        if word not in word_list:
            word_list.append(word)
    # 生成字典
    word_dict = {w:i for i,w in enumerate(word_list)} # 注意:单词是键,序号是值 
    # 词库大小:8
    vocab_size = len(word_list) 
    return word_sequence, word_list, word_dict, vocab_size

'''2-1:原版跳字模型'''
def Skip_Grams_original(sentences, word_sequence, word_dict):
    '''
    原版:不同语句之间的边界词,会组成(target,context)
    '''
    '''
    固定窗口大小(前后各一个词)
    依次把第2个词~倒数第2个词作为目标词,然后对于每个target依次选择前、后一个词作为上下文词.
    将每一对(目标词,上下文词)加入跳字模型
    '''
    skip_grams = []
    for i in range(1, len(word_sequence)-1):
        target = word_dict[word_sequence[i]] # 目标词序号键值
        context_1 = word_dict[word_sequence[i-1]] # 目标词的前一个词的序号
        context_2 = word_dict[word_sequence[i+1]] # 目标词的后一个词的序号
        # 添加(目标词,上下文词)样本对
        skip_grams.append([target, context_1]) 
        skip_grams.append([target, context_2])
    return skip_grams

'''2-1:跳字模型-cjk'''
def Skip_Grams(sentences, word_sequence, word_dict):
    '''
    对于原版进行了一些修改,当构建跳字模型时，
    仅在一个语句内挑选(target,context),不同句子之间不会产生输入输出对
    '''
    '''
    固定窗口大小(前后各一个词)
    依次把第2个词~倒数第2个词作为目标词,然后对于每个target依次选择前、后一个词作为上下文词.
    将每一对(目标词,上下文词)加入跳字模型
    '''
    skip_grams = []
    for i in range(len(sentences)):
        sentences_i_list = sentences[i].split(' ')
        length_sentences = len(sentences_i_list)
        for j in range(1,length_sentences-1):
            target = word_dict[sentences_i_list[j]] # 目标词序号键值
            context_1 = word_dict[sentences_i_list[j-1]] # 目标词的前一个词的序号
            context_2 = word_dict[sentences_i_list[j+1]] # 目标词的后一个词的序号
            skip_grams.append([target, context_1]) 
            skip_grams.append([target, context_2])
    return skip_grams

'''2-2:Word2Vec模型'''
class Word2Vec(nn.Module): # nn.Module是Word2Vec的父类
    def __init__(self): # (输入/输出层大小，嵌入层大小)
        '''
        super().__init__()
        继承父类的所有方法(),比如nn.Module的add_module()和parameters()
        '''
        super().__init__()

        '''2个全连接层
        troch.nn.Linear(in_features_size, out_features_size)
        输入向量:(batch_size, in_features_size)
        输出向量:(batch_size, out_features_size)
        '''
        self.W = nn.Linear(vocab_size, embedding_size, bias=False) # 输入层:vocab_size > embedding_size Weight
        self.WT = nn.Linear(embedding_size, vocab_size, bias=False) # 输出层:embedding_size > vocab_size Weight
        # Tips:W和WT不是转换关系

    def forward(self, X): # input_batch
        hidden_layer = self.W(X)  
        hidden_layer = hidden_layer.to(device)
        output_layer = self.WT(hidden_layer)
        output_layer = output_layer.to(device)
        ''' X --W--> hidden_layer --WT--> output_layer '''
        # X:[batch_size, vocab_size]
        # hidden_layer:[batch_size, embedding_size]
        # output_layer:[batch_size, vocab_size]        
        return output_layer

'''3-1:从跳字模型中随机抽样输入输出对(target,context),构建输入输出向量矩阵'''
def random_batch(skip_grams, batch_size, vocab_size): 
    input_batch = []
    target_batch = []
    random_index = np.random.choice(range(len(skip_grams)), batch_size, replace=False)
    """np.random.choice(L,num,replace=True):从数组/列表/元组中随机抽取num个元素
    L:数组/列表/元组
    num:选择元素的个数
    replace=True(Default)则可以取相同数字; replace=False则不能取相同数字
    """
    for i in random_index:
        # target
        # np.eye(8)生成8x8对角矩阵 -> one-hot向量
        input_batch.append(np.eye(vocab_size)[skip_grams[i][0]])  
        # context
        target_batch.append(skip_grams[i][1])
    """
    input_batch存储随机选择的target词向量,作为神经网络的输入
    target_batch存储对应的context单词,作为真实的输出
    由输入input_batch得到的输出output与目标值(真实输出)target_batch相比较,即可得到损失函数值
    """   
    input_batch = np.array(input_batch)   # (batch_size, vocab_size)
    target_batch = np.array(target_batch) # (batch_size)
    # type: array -> tensor
    '''input_batch包含batch_size个one-hot向量'''
    input_batch = torch.Tensor(input_batch)        # [batch_size, vocab_size]
    '''target_batch包含batch_size个context词的序号'''
    target_batch = torch.LongTensor(target_batch)  # [batch_size]

    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)    
    return input_batch, target_batch 
    """i.e.
    input_batch: 
        [[0. 1. 0. 0. 0. 0. 0. 0.],
        [0. 0. 0. 0. 0. 0. 0. 1.]] 
    target_batch:    
        [6 0] 
    即选出的输入输出对是(2,6)和(7,0)
    """

if __name__ == '__main__':
    device = ['cuda:0' if torch.cuda.is_available() else 'cpu'][0]
    embedding_size = 3  # 嵌入矩阵大小,即样本特征数,即嵌入向量的"长度"
    batch_size = 2      # 批量大小
    """batch_size:表示单次传递给程序用以训练的参数个数
    假设训练集有1000个数据,若设置batch_size=100,那么程序首先会用数据集第1-100个数据来训练模型。
    当训练完成后更新权重,再使用第101-200的个数据训练,用完训练集中的1000个数据后停止
    Pros:可以减少内存的使用、提高训练的速度(因为每次完成训练后都会更新权重值)
    Cons:使用少量数据训练时可能因为数据量较少而造成训练中的梯度值较大的波动。
    """

    chars = '*' * 20
    sentences = ["apple banana fruit", "banana orange fruit", "orange banana fruit",
                 "dog cat animal", "cat monkey animal", "monkey dog animal"]
    '''1.数据预处理'''
    word_sequence, word_list, word_dict, vocab_size = pre_process(sentences)

    '''2.构建模型'''

    '''2-1:构建跳字模型Skip_Grams''' # 上下一个词,[(target,context)......]
    skip_grams = Skip_Grams(sentences, word_sequence, word_dict)

    '''2-2:构建模型''' # Word2Vec():两个全连接层 input->hidden->output
    model = Word2Vec() # (输入/输出层大小，嵌入层大小)
    model.to(device)
    # 交叉熵损失函数,用于解决二分类或多分类问题,其内部会自动加上Softmax层
    criterion = nn.CrossEntropyLoss()
    # Adam动量法比随机梯度下降SGD更好
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    '''
    >>> model.parameters()) 
    (W): Linear(in_features=8, out_features=2, bias=False)
    (WT): Linear(in_features=2, out_features=8, bias=False)
    '''

    if os.path.exists('model_param.pt') == True:
        # 加载模型参数到模型结构
        model.load_state_dict(torch.load('model_param.pt', map_location=device))

    '''3.训练'''
    loss_record = []
    for epoch in range(50000):
        '''3-1:从跳字模型中随机抽样输入输出对(target,context),构建输入输出向量矩阵'''
        input_batch, target_batch = random_batch(skip_grams, batch_size, vocab_size)
        # print(input_batch, target_batch)
        
        '''3-2:导入模型进行(代码格式固定)'''
        # model->forward()
        output = model(input_batch) 
        output = output.to(device)
        optimizer.zero_grad() # 把梯度置零，即把loss关于weight的导数变成0.
        # output : [batch_size, vocab_size]
        # target_batch : [batch_size,] (LongTensor, not one-hot)
        loss = criterion(output, target_batch) # 将输出与真实目标值对比,得到损失值
        loss.backward() # 将损失loss向输入侧进行反向传播，梯度累计
        optimizer.step() # 根据优化器对W、b和WT、bT等参数进行更新(例如Adam动量法和随机梯度下降法SGD)
        
        if loss >= 0.01: # 连续30轮loss小于0.01则提前结束训练
            loss_record = []
        else:
            loss_record.append(loss.item())
            if len(loss_record) == 30:
                torch.save(model.state_dict(), 'model_param.pt')
                break    

        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'Loss = {:.6f}'.format(loss))
            torch.save(model.state_dict(), 'model_param.pt')  
    
    W, WT = model.parameters() # W即为嵌入矩阵

    '''4.Draw'''
    print(input_batch)
    print(W, WT)
    # 根据嵌入矩阵,可视化各个单词的特征值
    for i, word in enumerate(word_list): # 枚举词库中每个单词
        x, y = W[0][i].item(), W[1][i].item()
        plt.scatter(x, y) # 散点图
        plt.annotate(word, xy=(x,y))
        '''plt.annotate(s='str',xy=(x,y)......)函数用于标注文字
        s:文本注释内容
        xy:被注释的坐标点
        color 设置字体颜色 color={'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'}
        '''
    plt.show()
