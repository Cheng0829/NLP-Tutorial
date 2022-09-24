"""
Task: 基于TextLSTM的单词字母预测
Author: ChengJunkai @github.com/Cheng0829
Email: chengjunkai829@gmail.com
Date: 2022/09/09
Reference: Tae Hwan Jung(Jeff Jung) @graykode
"""

import numpy as np
import torch, os, sys, time
import torch.nn as nn
import torch.optim as optim

'''1.数据预处理'''
def pre_process(seq_data):
    char_arr = [c for c in 'abcdefghijklmnopqrstuvwxyz']
    word_dict = {n:i for i, n in enumerate(char_arr)}
    number_dict = {i:w for i, w in enumerate(char_arr)}
    # 字母类别数:26,即嵌入向量维度
    n_class = len(word_dict)  # number of class(=number of vocab)
    return char_arr, word_dict, number_dict, n_class

'''根据句子数据,构建词元的嵌入向量及目标词索引'''
def make_batch(seq_data):
    input_batch, target_batch = [], []
    # 每个样本单词
    for seq in seq_data: 
        # e.g. input : ['m','a','k'] -> [num1, num2, num3]
        input = [word_dict[n] for n in seq[:-1]] 
        target = word_dict[seq[-1]] # e.g. 'e' is target
        input_batch.append(np.eye(n_class)[input])
        target_batch.append(target)
    '''input_batch : [batch_size(样本数), n_step(样本单词数), n_class] -> [10, 3, 26]'''
    input_batch = torch.FloatTensor(np.array(input_batch)) 
    # print(input_batch.shape)
    target_batch = torch.LongTensor(np.array(target_batch))
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    return input_batch, target_batch

'''2.构建模型:LSTM(本实验结构图详见笔记)'''
class TextLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        # n_class是字母类别数(26),即嵌入向量维度
        self.lstm = nn.LSTM(input_size=n_class, hidden_size=hidden_size)
        self.W = nn.Linear(hidden_size, n_class, bias=False)
        self.b = nn.Parameter(torch.ones([n_class]))

    '''每个样本输入的单词数和模型的时间步长度相等'''
    def forward(self, X):
        # X : [batch_size, n_step, n_class] [10, 3, 26] 
        # input : [n_step, batch_size, n_class] [3, 10, 26]
        # input : [输入序列长度(时间步长度),样本数,嵌入向量维度]
        '''transpose转置: [10, 3, 26] -> [3, 10, 26]'''
        input = X.transpose(0, 1)
        # hidden_state:[num_layers*num_directions, batch_size, hidden_size]
        # hidden_state:[层数*网络方向,样本数,隐藏层的维度(隐藏层神经元个数)]
        hidden_state = torch.zeros(1, len(X), hidden_size)  
        hidden_state = hidden_state.to(device)
        # cell_state:[num_layers*num_directions, batch_size, hidden_size]
        # cell_state:[层数*网络方向,样本数,隐藏层的维度(隐藏层神经元个数)]
        cell_state = torch.zeros(1, len(X), hidden_size)     
        cell_state = cell_state.to(device)
        '''
        一个LSTM细胞单元有三个输入,分别是$输入向量x^{<t>}、隐藏层向量a^{<t-1>}
        和记忆细胞c^{<t-1>}$;一个LSTM细胞单元有三个输出,分别是$输出向量y^{<t>}、
        隐藏层向量a^{<t>}和记忆细胞c^{<t>}$
        '''
        # outputs:[3,10,128] final_hidden_state:[1,10,128] final_cell_state:[1,10,128])
        outputs, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))
        outputs = outputs.to(device)
        '''取最后一个单元的隐藏层激活状态输出值'''
        '''既可以用outputs[-1],也可以用final_hidden_state[0]'''
        final_output = outputs[-1]  # [batch_size, hidden_size]
        Y_t = self.W(final_output) + self.b  # Y_t : [batch_size, n_class]
        return Y_t


if __name__ == '__main__':
    hidden_size = 128 # number of hidden units in one cell
    device = ['cuda:0' if torch.cuda.is_available() else 'cpu'][0]
    seq_data = ['make', 'need', 'coal', 'word', 'love', 'hate', 'live', 'home', 'hash', 'star']

    '''1.数据预处理'''
    char_arr, word_dict, number_dict, n_class = pre_process(seq_data)
    input_batch, target_batch = make_batch(seq_data)

    '''2.构建模型'''
    model = TextLSTM()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if os.path.exists('model_param.pt') == True:
        # 加载模型参数到模型结构
        model.load_state_dict(torch.load('model_param.pt', map_location=device))

    '''3.训练'''
    print('{}\nTrain\n{}'.format('*'*30, '*'*30))
    loss_record = []
    for epoch in range(1000):
        optimizer.zero_grad()
        # X : [batch_size, n_step, n_class]
        output = model(input_batch)
        output = output.to(device)
        loss = criterion(output, target_batch)
        loss.backward()
        optimizer.step()    
        if loss >= 0.001: # 连续30轮loss小于0.01则提前结束训练
            loss_record = []
        else:
            loss_record.append(loss.item())
            if len(loss_record) == 30:
                torch.save(model.state_dict(), 'model_param.pt')
                break    

        if ((epoch+1) % 100 == 0):
            print('Epoch:', '%04d' % (epoch + 1), 'Loss = {:.6f}'.format(loss))
            torch.save(model.state_dict(), 'model_param.pt')

    '''4.预测'''
    print('{}\nTest\n{}'.format('*'*30, '*'*30))
    inputs = ['mak','ma','look'] # make look
    for input in inputs: # 每个样本逐次预测,避免长度不同
        input_batch, target_batch = make_batch([input])
        predict = model(input_batch).data.max(1, keepdim=True)[1]
        print(input + ' -> ' + input + number_dict[predict.item()])

'''
为什么训练集输入字母数都是3,但是测试集可以不是?
make_batch的input_batch维度是[batch_size(样本数), n_step(样本单词数),n_class(26)]
n_step是输入序列长度,之前疑惑为什么只有3个lstm单元,却可以输入其他个数的字母?

    实际上,模型并没有把时间步作为一个超参数,也就是时间步随输入样本而变化,在训练集中,n_step均为3,
    但是,在测试集中,三个单词都是分别作为样本集输入的,也就是时间步分别为3,2,4,
    最后在self.lstm(input, (hidden_state, cell_state))中,模型会自动根据input的序列长度,分配时间步
    
    但由于是一次性输入一个样本集,所以样本集中各个样本长度必须一致,否则报错
    因此必须把预测的inputs中各个单词分别放进容量为1的样本集单独输入

    需要指出的是,由于模型训练的是根据3个字母找到最后以1个字母,
    所以如果长度不匹配,即使单词在训练集中,
    也不能取得好的结果,比如"ma"的预测结果并不一定是训练集中的"mak"
'''
