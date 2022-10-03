"""
Task: 基于Seq2Seq和注意力机制的句子翻译
Author: ChengJunkai @github.com/Cheng0829
Email: chengjunkai829@gmail.com
Date: 2022/09/13
Reference: Tae Hwan Jung(Jeff Jung) @graykode
"""

from tkinter import font
import numpy as np
import torch, time, os, sys
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# S: 表示开始进行解码输入的符号。
# E: 表示结束进行解码输出的符号。
# P: 当前批次数据大小小于时间步长时将填充空白序列的符号

'''1.数据预处理'''
def pre_process(sentences):
    # 分词
    word_sequence = " ".join(sentences).split()
    # 去重
    word_list = []
    '''
    如果用list(set(word_sequence))来去重,得到的将是一个随机顺序的列表(因为set无序),
    这样得到的字典不同,保存的上一次训练的模型很有可能在这一次不能用
    (比如上一次的模型预测碰见我:0,,就输出i:7,但这次模型i在字典8号位置,也就无法输出正确结果)
    '''
    for word in word_sequence:
        if word not in word_list:
            word_list.append(word)
    word_dict = {w:i for i, w in enumerate(word_list)}
    number_dict = {i:w for i, w in enumerate(word_list)}
    # 词库大小,也是嵌入向量维度
    n_class = len(word_dict)  # 12
    return word_list, word_dict, number_dict, n_class 

'''根据句子数据,构建词元的嵌入向量'''
def make_batch(sentences,word_dict):
    # [1, 6, 12] [样本数, 输入句子长度, 嵌入向量维度(单词类别数)]
    input_batch = [np.eye(n_class)[[word_dict[n] for n in sentences[0].split()]]]
    # [1, 5, 12] [样本数, 输出句子长度, 嵌入向量维度(单词类别数)]
    output_batch = [np.eye(n_class)[[word_dict[n] for n in sentences[1].split()]]]
    # [1, 5] [样本数, 输出句子长度]
    target_batch = [[word_dict[n] for n in sentences[2].split()]]

    input_batch = torch.FloatTensor(np.array(input_batch)).to(device)
    output_batch =torch.FloatTensor(np.array(output_batch)).to(device)
    target_batch = torch.LongTensor(np.array(target_batch)).to(device)

    return input_batch, output_batch, target_batch

'''2.构建模型'''
class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.encoder_cell = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5)
        self.decoder_cell = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5)
        # Linear for attention
        self.attn = nn.Linear(n_hidden, n_hidden)
        self.out = nn.Linear(2*n_hidden, n_class)

    '''output, _ = model(input_batch, hidden_0, output_batch)'''
    def forward(self, encoder_inputs, hidden_0, decoder_inputs):
        # [6, 1, 12] [输入句子长度(n_step), 样本数, 嵌入向量维度(单词类别数)]
        encoder_inputs = encoder_inputs.transpose(0, 1)  # encoder_inputs: [n_step(=n_step, time step), batch_size, n_class]
        # [5, 1, 12] [输出句子长度(n_step), 样本数, 嵌入向量维度(单词类别数)]
        decoder_inputs = decoder_inputs.transpose(0, 1)  # decoder_inputs: [n_step(=n_step, time step), batch_size, n_class]
        # print(encoder_inputs.shape, decoder_inputs.shape)
        '''编码器encoder'''
        # encoder_outputs : [实际的n_step, batch_size, num_directions(=1)*n_hidden] # [5,1,128]
        # encoder_states : [num_layers*num_directions, batch_size, n_hidden] # [1,1,128]
        '''encoder_states是最后一个时间步的输出(即隐藏层状态),和encoder_outputs的最后一个元素一样'''
        encoder_outputs, encoder_states = self.encoder_cell(encoder_inputs, hidden_0)
        encoder_outputs = encoder_outputs # [6,1,128]
        encoder_states = encoder_states # [1,1,128]
        # print(encoder_outputs.shape, encoder_states.shape)
        n_step = len(decoder_inputs) # 5
        # 返回一个未初始化的张量,内部均为随机数
        output = torch.empty([n_step, 1, n_class]).to(device) # [5,1,12]
        
        '''获取注意力权重 : between(整个编码器上的隐状态, 整个解码器上的隐状态)
        有两次加权求和,一次是bmm,一次是dot,对应两个for循环
        '''
        trained_attn = []
        '''解码器上的每个时间步'''
        for i in range(n_step): # 5
            '''解码器'''
            '''decoder_inputs[i]即只需要第i个时间步上面的解码器输入,但必须是三维,所以用unsqueeze升一维'''
            decoder_input_one = decoder_inputs[i].unsqueeze(0) # 升维
            '''decoder_output_one 和 encoder_states 其实是一样的 因为decoder_cell只算了一个时间步'''
            decoder_output_one, encoder_states = self.decoder_cell(decoder_input_one, encoder_states)
            decoder_output_one = decoder_output_one
            encoder_states = encoder_states
            '''attn_weights是一个解码器时间步隐状态和整个编码器之间的注意力权重'''
            # attn_weights : [1, 1, n_step] # [1,1,6]
            attn_weights = self.get_attn_one_to_all(decoder_output_one, encoder_outputs)
            # 
            '''squeeze():[1,1,6]->[6,] data:只取数据部分,剔除梯度部分 numpy:转换成一维矩阵'''
            trained_attn.append(attn_weights.squeeze().data.numpy())
            # numpy遍历不能存在于cuda,所以必须先作为cpu变量进行操作,再进行转换
            attn_weights = attn_weights.to(device) 
            """a.bmm(b)和torch.bmm(a,b)一样
                a:(z,x,y)
                b:(z,y,c)
                则result = torch.bmm(a,b),维度为:(z,x,c)
            """
            '''利用attn第i时刻Encoder的隐状态的加权求和,得到上下文向量,即融合了注意力的模型输出'''
            # context:[1,1,n_step(=5)]x[1,n_step(=5),n_hidden(=128)]=[1,1,128]
            context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
            # decoder_output_one : [batch_size(=1), num_directions(=1)*n_hidden]
            decoder_output_one = decoder_output_one.squeeze(0) # [1,1,128] -> [1,128]
            # [1, num_directions(=1)*n_hidden] # [1,128]
            context = context.squeeze(1)  
            '''把上下文向量和解码器隐状态进行concat,得到融合了注意力的模型输出'''
            # torch.cat的dim=1代表在第二个维度上拼接 ,所以[1,128]+[1,128]->[1,256]
            # output[i] = self.out(torch.cat((decoder_output_one, context), 1))
            output[i] = self.out(torch.cat((decoder_output_one, context), 1))
        # output: [5,1,12] -> [1,5,12] -> [5,12]
        return output.transpose(0, 1).squeeze(0), np.array(trained_attn)

    '''获取注意力权重 : between(解码器的一个时间步的隐状态, 整个编码器上的隐状态)'''
    def get_attn_one_to_all(self, decoder_output_one, encoder_outputs):  
        n_step = len(encoder_outputs) # 6
        attn_scores = torch.zeros(n_step)  # attn_scores : [n_step,] -> [6,]
        
        '''对解码器的每个时间步获取注意力权重'''
        for i in range(n_step):
            encoder_output_one = encoder_outputs[i]
            attn_scores[i] = self.get_attn_one_to_one(decoder_output_one, encoder_output_one)

        """F.softmax(matrix,dim) 将scores标准化为0到1范围内的权重
            softmax(x_i) = exp(x_i) / sum( exp(x_1) + ··· + exp(x_n) )
            由于attn_scores是一维张量,所以F.softmax不用指定dim
        """
        # .view(1,1,-1)把所有元素都压到最后一个维度上,把一维的张量变成三维的
        return F.softmax(attn_scores).view(1, 1, -1) # [6,] -> [1,1,6]

    '''获取注意力权重 : between(编码器的一个时间步的隐状态, 解码器的一个时间步的隐状态)'''
    def get_attn_one_to_one(self, decoder_output_one, encoder_output_one):  
        # decoder_output_one : [batch_size, num_directions(=1)*n_hidden] # [1,128]
        # encoder_output_one : [batch_size, num_directions(=1)*n_hidden] # [1,128]
        # score : [batch_size, n_hidden] -> [1,128]
        score = self.attn(encoder_output_one)  
        '''X.view(shape) 
        >>> X = torch.ones((3,2))
        >>> X = X.view(2,3) # X形状变为(2,3)
        >>> X = X.view(-1) # X形状变为一维
        '''
        # decoder_output_one : [n_step(=1), batch_size(=1), num_directions(=1)*n_hidden] -> [1,1,128]
        # score : [batch_size, n_hidden] -> [1,128]
        # 求点积
        return torch.dot(decoder_output_one.view(-1), score.view(-1))  # inner product make scalar value

def translate(sentences):
    input_batch, output_batch, target_batch = make_batch(sentences,word_dict)
    blank_batch = [np.eye(n_class)[[word_dict[n] for n in 'SPPPP']]]
    # test_batch: [1,5,12] [batch_size,len_sen,dict_size]
    test_batch = torch.FloatTensor(np.array(blank_batch)).to(device) 
    dec_inputs = torch.FloatTensor(np.array(blank_batch)).to(device) 

    '''贪婪搜索'''
    for i in range(len(test_batch[0])):
        # predict: [len_sen, dict_size] [5,12]
        predict, trained_attn = model(input_batch, hidden_0, dec_inputs) 
        predict = predict.data.max(1, keepdim=True)[1] # [5,1] [sen_len,1]
        # 覆盖之前的padding字符
        dec_inputs[0][i][word_dict['P']] = 0
        dec_inputs[0][i][predict[i][0]] = 1
        
    predict, trained_attn = model(input_batch, hidden_0, dec_inputs) 
    predict = predict.data.max(1, keepdim=True)[1] # [5,1] [sen_len,1]
    decoded = [word_list[i] for i in predict]
    real_decoded = decoded # 记录不清除特殊字符的decoded

    '''清除特殊字符'''
    '''训练集的target均以E结尾,所以模型输出最后一个值也会是E'''
    if 'E' in decoded:
        end = decoded.index('E') # 5
        decoded = decoded[:end] # 删除结束符及之后的所有字符
    else:
        return # 报错
    while(True):
        if 'P' in decoded:
            del decoded[decoded.index('P')] # 删除空白符
        else:
            break

    # 把列表元素合成字符串
    translated = ' '.join(decoded) 
    real_output = ' '.join(real_decoded) 
    return translated, real_output

if __name__ == '__main__':
    # n_step = 5 # number of cells(= number of Step)
    chars = 30 * '*'
    n_hidden = 128 # number of hidden units in one cell
    '''GPU比CPU慢的原因大致为:
    数据传输会有很大的开销,而GPU处理数据传输要比CPU慢,
    而GPU在矩阵计算上的优势在小规模神经网络中无法明显体现出来
    '''
    device = ['cuda:0' if torch.cuda.is_available() else 'cpu'][0]
    sentences = ['我 想 喝 啤 酒 P', 'S i want a beer', 'i want a beer E']

    '''1.数据预处理'''
    word_list, word_dict, number_dict, n_class = pre_process(sentences)
    input_batch, output_batch, target_batch = make_batch(sentences,word_dict)
    # hidden_0 : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
    hidden_0 = torch.zeros(1, 1, n_hidden).to(device) # [1,1,128]

    '''2.构建模型'''
    model = Attention()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if os.path.exists('model_param.pt') == True:
        # 加载模型参数到模型结构
        model.load_state_dict(torch.load('model_param.pt', map_location=device))

    '''3.训练'''
    print('{}\nTrain\n{}'.format('*'*30, '*'*30))
    loss_record = []
    for epoch in range(1000):
        optimizer.zero_grad()
        output, trained_attn = model(input_batch, hidden_0, output_batch)
        output = output.to(device)
        loss = criterion(output, target_batch.squeeze(0)) # .squeeze(0)降成1维
        loss.backward()
        optimizer.step()

        if loss >= 0.0001: # 连续30轮loss小于0.01则提前结束训练
            loss_record = []
        else:
            loss_record.append(loss.item())
            if len(loss_record) == 30:
                torch.save(model.state_dict(), 'model_param.pt')
                break    

        if (epoch + 1) % 100 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'Loss = {:.6f}'.format(loss))
            torch.save(model.state_dict(), 'model_param.pt')

    '''4.测试'''
    print('{}\nTest\n{}'.format('*'*30, '*'*30))
    input = sentences[0]
    output, real_output = translate(input)
    print(sentences[0].replace(' P', ''), '->', output)

    '''5.可视化注意力权重矩阵'''
    trained_attn = trained_attn.round(2)
    fig = plt.figure(figsize=(len(input.split()), len(real_output.split()))) # (5,5)
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(trained_attn, cmap='viridis')
    ax.set_xticklabels([''] + input.split(), \
        fontdict={'fontsize': 14}, fontproperties='SimSun') # 宋体
    ax.set_yticklabels([''] + real_output.split(), \
        fontdict={'fontsize': 14}, fontproperties='SimSun')
    plt.show()
