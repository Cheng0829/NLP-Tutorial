"""
Task: 基于Bi-LSTM和注意力机制的文本情感分类
Author: ChengJunkai @github.com/Cheng0829
Email: chengjunkai829@gmail.com
Date: 2022/09/14
"""

import numpy as np
import torch,time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

'''1.数据预处理'''
def pre_process(sentences):
    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w:i for i, w in enumerate(word_list)}
    word_dict["''"] = len(word_dict)
    word_list = word_list.append("''")
    vocab_size = len(word_dict) # 词库大小16
    max_size = 0
    for sen in sentences:
        if len(sen.split()) > max_size:
            max_size = len(sen.split()) # 最大长度3
    for i in range(len(sentences)):
        if len(sentences[i].split()) < max_size:
            sentences[i] = sentences[i] + " ''" * (max_size - len(sentences[i].split()))
    
    return sentences, word_list, word_dict, vocab_size, max_size

def make_batch(sentences):
    # 对于每个句子,返回包含句子内每个单词序号的列表
    inputs = [np.array([word_dict[n] for n in sen.split()]) for sen in sentences] # [6,3]
    targets = [out for out in labels]
    #print(inputs)
    inputs = torch.LongTensor(np.array(inputs)).to(device)
    targets = torch.LongTensor(np.array(targets)).to(device)
    '''情感分类构建嵌入矩阵,没有eye()'''
    return inputs, targets

class BiLSTM_Attention(nn.Module):
    def __init__(self):
        super(BiLSTM_Attention, self).__init__()
        '''情感分类构建嵌入矩阵,没有eye()'''
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, n_hidden, bidirectional=True)
        self.out = nn.Linear(2*n_hidden, num_classes)

    def forward(self, X):
        # input : [batch_size, n_step, embedding_dim] [6,3,2]
        input = self.embedding(X) 
        # input : [n_step, batch_size, embedding_dim] [3,6,2]
        # input : [输入序列长度(时间步长度),样本数,嵌入向量维度]
        input = input.permute(1, 0, 2) 
        # hidden_state : [num_layers(=1)*num_directions(=2), batch_size, n_hidden]
        # hidden_state : [层数*网络方向,样本数,隐藏层的维度(隐藏层神经元个数)]
        hidden_state = torch.zeros(1*2, len(X), n_hidden).to(device) 
        # cell_state : [num_layers*num_directions, batch_size, hidden_size]
        # cell_state : [层数*网络方向,样本数,隐藏层的维度(隐藏层神经元个数)]
        cell_state = torch.zeros(1*2, len(X), n_hidden).to(device) 
        # final_hidden_state, final_cell_state : [num_layers(=1)*num_directions(=2), batch_size, n_hidden]
        ltsm_output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))
        # ltsm_output : [batch_size, n_step, n_hidden*num_directions(=2)]
        ltsm_output = ltsm_output.permute(1, 0, 2) 
        attn_output, attention = self.attention_net(ltsm_output, final_hidden_state)
        # model : [batch_size, num_classes], attention : [batch_size, n_step]
        return self.out(attn_output), attention 

    '''两次bmm加权求和,相当于两次for循环''' 
    # lstm_output : [batch_size, n_step, n_hidden*num_directions(=2)] [6,3,16]
    # final_hidden_state : [num_layers(=1)*num_directions(=2), batch_size, n_hidden] [2,6,8]
    def attention_net(self, lstm_output, final_hidden_state):
        # final_hidden_state : [batch_size, n_hidden*num_directions(=2), 1(=n_layer)] [6,16,1]
        final_hidden_state = final_hidden_state.view(-1, 2*n_hidden, 1) 

        '''第一次bmm加权求和:: lstm_output和final_hidden_state生成注意力权重attn_weights'''
        # [6,3,16]*[6,16,1] -> [6,3,1] -> attn_weights : [batch_size, n_step] [6,3]
        attn_weights = torch.bmm(lstm_output, final_hidden_state).squeeze(2) # 第3维度降维
        softmax_attn_weights = F.softmax(attn_weights, 1) # 按列求值 [6,3]

        '''第二次bmm加权求和 : lstm_output和注意力权重attn_weights生成上下文向量context,即融合了注意力的模型输出'''
        # [batch_size, n_hidden*num_directions, n_step] * [batch_size,n_step,1] \
        # = [batch_size, n_hidden*num_directions, 1] : [6,16,3] * [6,3,1] -> [6,16,1] -> [6,16]
        context = torch.bmm(lstm_output.transpose(1, 2), softmax_attn_weights.unsqueeze(2)).squeeze(2)
        softmax_attn_weights = softmax_attn_weights.to('cpu') # numpy变量只能在cpu上
        
        '''各个任务求出context之后的步骤不同,LSTM的上下文不需要和Seq2Seq中的一样和decoder_output连接'''
        return context, softmax_attn_weights.data.numpy() 

if __name__ == '__main__':
    chars = 30 * '*'
    embedding_dim = 3 # embedding size
    n_hidden = 8  # number of hidden units in one cell
    num_classes = 2  # 0 or 1
    '''GPU比CPU慢的原因大致为:
    数据传输会有很大的开销,而GPU处理数据传输要比CPU慢,
    而GPU在矩阵计算上的优势在小规模神经网络中无法明显体现出来
    '''
    # device = ['cuda:0' if torch.cuda.is_available() else 'cpu'][0]
    device = 'cpu'
    # 3 words sentences (=sequence_length is 3)
    sentences = ["i love you", "he loves me", "don't leave", \
                 "i hate you", "sorry for that", "this is awful"]
    labels = [1, 1, 1, 0, 0, 0]  # 1 is good, 0 is not good.

    '''1.数据预处理'''
    sentences, word_list, word_dict, vocab_size, max_size = pre_process(sentences)
    
    '''2.构建模型'''
    model = BiLSTM_Attention()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    inputs, targets = make_batch(sentences)

    '''3.训练'''
    old = time.time()
    print(chars)
    print('Train')
    print(chars)
    for epoch in range(1000):
        optimizer.zero_grad()
        output, attention = model(inputs)
        output = output.to(device)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 1000 == 0:
            new = time.time()
            print('Epoch:%d\n'%(epoch+1), 'Cost=%.6f'%loss, 'Time=%.3fs'%(new-old))
            old = time.time()

    '''4.预测'''
    print(chars)
    print('Predict')
    print(chars)
    test_text = 'sorry i love you' # 'sorry hate you'
    # 返回包含每个单词序号的列表矩阵(为了有2个维度,还要加一个中括号升维)
    tests = [np.array([word_dict[n] for n in test_text.split()])]
    test_batch = torch.LongTensor(np.array(tests)).to(device)
    predict, attn_test = model(test_batch)
    predict = predict.data.max(1, keepdim=True)[1]
    print('The emotion of "%s" is '%test_text, end='')
    if predict[0][0] == 0:
        print('bad!')
    else:
        print('good!')

    '''5.可视化注意力权重矩阵'''
    fig = plt.figure(figsize=(0.5*len(sentences), 0.5*len(sentences[0]))) # [batch_size, n_step]
    ax = fig.add_subplot(1, 1, 1)
    # attention : (6, 3)
    ax.matshow(attention, cmap='viridis')
    word_show = ['单词'] * len(sentences[0])
    word_show = [word_show[i] + str(i+1) for i in range(len(sentences[0]))] # ['word_1', 'word_2', 'word_3']
    ax.set_xticklabels([''] + word_show, fontdict={'fontsize': 14} , fontproperties='SimSun')
    sentence_show = ['句子'] * len(sentences)
    sentence_show = [sentence_show[i] + str(i+1) for i in range(len(sentence_show))] # ['sentence_1', 'sentence_2', 'sentence_3', 'sentence_4', 'sentence_5', 'sentence_6']
    ax.set_yticklabels([''] + sentence_show, fontdict={'fontsize': 14}, fontproperties='SimSun')
    plt.show()
