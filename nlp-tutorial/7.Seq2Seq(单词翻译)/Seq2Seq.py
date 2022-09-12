"""
Task: 基于Seq2Seq的单词翻译
Author: ChengJunkai @github.com/Cheng0829
Date: 2022/09/11
"""

import numpy as np
import torch,time
import torch.nn as nn

# S: Symbol that shows starting of decoding input
# E: Symbol that shows ending of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is short than time steps
# S: 表示开始进行解码输入的符号。
# E: 表示结束进行解码输出的符号。
# P: 当前批次数据大小小于时间步长时将填充空白序列的符号


'''1.数据预处理'''
def pre_process(chars):
    char_arr = [char for char in chars]
    # 字符字典
    num_dict = {n:i for i,n in enumerate(char_arr)}    
    # 字符种类
    n_class = len(num_dict)
    # 样本数
    batch_size = len(seq_data)
    return char_arr, num_dict, n_class, batch_size

'''4.预测'''
def translate(input_word):
    input_batch, output_batch = [], []
    # 把每个单词补充到时间步长度
    input_word = input_word + 'P' * (n_step - len(input_word))
    # 换成序号
    input = [num_dict[n] for n in input_word] # 
    # 除了一个表示开始解码输入的符号,其余均为空白符号
    output = [num_dict[n] for n in 'S'+'P'*n_step]

    input_batch = np.eye(n_class)[input]
    output_batch = np.eye(n_class)[output]

    input_batch = torch.FloatTensor(np.array(input_batch)).unsqueeze(0).to(device)
    output_batch = torch.FloatTensor(np.array(output_batch)).unsqueeze(0).to(device)
    '''样本集为1'''
    # hidden : [num_layers*num_directions, batch_size, n_hidden] [1,1,128]
    hidden = torch.zeros(1, 1, n_hidden).to(device)
    '''output : [n_step+1(=6), batch_size, n_class] [6,1,29]'''
    output = model(input_batch, hidden, output_batch) # [6,1,29]
    
    '''torch.tensor.data.max(dim,keepdim) 用于找概率最大的输出值及其索引
    Args:
        dim (int): 在哪一个维度求最大值
        keepdim (Boolean): 保持维度. 
            keepdim=True:当tensor维度>1时,得到的索引和输出值仍然保持原来的维度
            keepdim=False:当tensor维度>1时,得到的索引和输出值为1维
    '''
    '''dim=2:在第2维求最大值  [1]:只需要索引'''
    predict = output.data.max(2, keepdim=True)[1] # select n_class dimension
    '''由于predict中元素全为索引整数,所以即使有几个中括号,仍可以直接作为char_arr的索引'''
    decoded = [char_arr[i] for i in predict] # ['m', 'e', 'n', 'P', 'P', 'E']

    '''清除特殊字符'''
    '''训练集的target均以E结尾,所以模型输出最后一个值也会是E'''
    if 'E' in decoded:
        end = decoded.index('E') # 5
        del decoded[end] # 删除结束符
    while(True):
        if 'P' in decoded:
            del decoded[decoded.index('P')] # 删除空白符
        else:
            break

    # 把列表元素合成字符串
    translated = ''.join(decoded) 
    return translated

'''根据句子数据,构建词元的嵌入向量及目标词索引'''
def make_batch(seq_data):
    input_batch, output_batch, target_batch = [], [], []

    for seq in seq_data:
        for i in range(2):
            # 把每个单词补充到时间步长度
            seq[i] = seq[i] + 'P' * (n_step - len(seq[i]))

        input = [num_dict[n] for n in seq[0]]
        # output是decoder的输入,所以加上开始解码输入的符号
        output = [num_dict[n] for n in ('S' + seq[1])]
        # target是decoder的输出,所以加上开始解码输出的符号
        target = [num_dict[n] for n in (seq[1] + 'E')]

        input_batch.append(np.eye(n_class)[input])
        output_batch.append(np.eye(n_class)[output])
        target_batch.append(target) # not one-hot

    '''input_batch用于编码器输入, output_batch用于解码器输入, target_batch用于比较计算误差'''
    # [样本数,时间步长度,嵌入向量维度] -> [6,5,29] 
    input_batch = torch.FloatTensor(np.array(input_batch)).to(device) 
    # [样本数,时间步长度+1,嵌入向量维度] -> [6,6,29] 
    output_batch = torch.FloatTensor(np.array(output_batch)).to(device) 
    # [样本数,时间步长度+1] -> [6,6]
    target_batch = torch.LongTensor (np.array(target_batch)).to(device)  
 
    return input_batch, output_batch, target_batch


'''2.构建模型'''
class Seq2Seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5)
        self.decoder = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5)
        self.fc = nn.Linear(n_hidden, n_class)

    '''编码器5个时间步,解码器六个:一个时间步对应一个单词字母'''
    def forward(self, encoder_input, encoder_hidden, decoder_input):
        '''
            encoder_input: input_batch
            encoder_hidden: hidden
            decoder_input: output_batch        
        '''
        # encoder_input: [n_step, batch_size, n_class] -> [5,6,29]
        encoder_input = encoder_input.transpose(0, 1)
        # decoder_input: [n_step, batch_size, n_class] -> [6,6,29]
        decoder_input = decoder_input.transpose(0, 1)

        '''编码器输出作为解码器输入的hidden'''
        # hidden最后只从一个单元里输出,所以第一维是1
        # encoder_states : [num_layers(=1)*num_directions(=1), batch_size, n_hidden] # [1,6,128]
        _, encoder_states = self.encoder(encoder_input, encoder_hidden)
        encoder_states = encoder_states.to(device)
        '''解码器输出'''
        # outputs : [n_step+1(=6), batch_size, num_directions(=1)*n_hidden(=128)] # [6,6,128]
        outputs, _ = self.decoder(decoder_input, encoder_states)
        outputs = outputs.to(device)
        '''全连接层'''
        # output : [n_step+1(=6), batch_size, n_class]
        output = self.fc(outputs) # [6,6,29]
        return output

if __name__ == '__main__':
    chars_print = '*' * 30
    n_step = 5 # (样本单词均不大于5,所以n_step=5)
    n_hidden = 128
    device = ['cuda:0' if torch.cuda.is_available() else 'cpu'][0]
    # 单词序列
    seq_data = [['man', 'men'], ['black', 'white'], ['king', 'queen'], \
                ['girl', 'boy'], ['up', 'down'], ['high', 'low']]
    chars = 'SEPabcdefghijklmnopqrstuvwxyz'

    '''1.数据预处理'''
    char_arr, num_dict, n_class, batch_size = pre_process(chars)

    '''2.构建模型'''
    model = Seq2Seq()
    model.to(device)
    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    '''根据句子数据,构建词元的嵌入向量及目标词索引'''
    input_batch, output_batch, target_batch,  = make_batch(seq_data)

    '''3.训练'''
    old = time.time()
    print(chars_print)
    print('Train')
    print(chars_print)
    for epoch in range(10000):
        # make hidden shape [num_layers * num_directions, batch_size, n_hidden] [1,6,128]
        hidden = torch.zeros(1, batch_size, n_hidden)
        hidden = hidden.to(device)

        optimizer.zero_grad()
        # input_batch : [样本数, 时间步长度, 嵌入向量维度]
        # output_batch : [样本数, 时间步长度+1, 嵌入向量维度]
        # target_batch : [样本数, 时间步长度+1] 
        output = model(input_batch, hidden, output_batch) # [6,6,29]
        # output : [max_len+1, batch_size, n_class]
        output = output.transpose(0, 1) # [batch_size, max_len+1(=6), n_class] [6,6,29]
        
        '''
        criterion的输入应该是output二维,target_batch一维,此实验不是这样,
        一个单词样本分为几个字母,每个字母指定一个字母输出,因此target_batch是二维
        所以要遍历相加.
        '''
        loss = 0
        for i in range(0, len(target_batch)):
            '''output: [6,6,29] target_batch:[6,6]'''
            loss = loss + criterion(output[i], target_batch[i])
        # print(loss)
        # print(criterion(output, target_batch))
        loss.backward()
        optimizer.step() 
        if (epoch + 1) % 1000 == 0:
            new = time.time()
            print('Epoch:%d\n' % (epoch + 1), 'Cost={:.6f}'.format(loss), end=' ')
            print('Time=%.3fs'%(new-old))
            old = time.time()

    '''4.预测'''
    test_words = ['man','men','king','black','upp']
    print(chars_print)
    print('test')
    print(chars_print)
    for word in test_words:
        print('%s ->'%word, translate('man'))
