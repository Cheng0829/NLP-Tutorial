"""
Task: 基于Transformer的句子翻译
Author: ChengJunkai @github.com/Cheng0829
Email: chengjunkai829@gmail.com
Date: 2022/09/17
Reference: Tae Hwan Jung(Jeff Jung) @graykode
"""

import numpy as np
import torch, time, itertools, os, sys 
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# S: 表示开始进行解码输入的符号.
# E: 表示结束进行解码输出的符号.
# P: 当前批次数据大小小于时间步长时将填充空白序列的符号

'''1.数据预处理'''
def pre_process(sentences):
    # P在第一个,方便处理
    src_sequence = ['P']
    src_sequence.extend(sentences[0].split())
    src_list = []
    '''
    如果用list(set(word_sequence))来去重,得到的将是一个随机顺序的列表(因为set无序),
    这样得到的字典不同,保存的上一次训练的模型很有可能在这一次不能用
    (比如上一次的模型预测碰见i:0,love:1,就输出you:2,但这次模型you在字典3号位置,也就无法输出正确结果)
    '''
    for word in src_sequence:
        if word not in src_list:
            src_list.append(word)
    src_dict = {w:i for i,w in enumerate(src_list)}
    src_dict_size = len(src_dict)
    src_len = len(sentences[0].split()) # length of source

    # P在第一个,方便处理
    tgt_sequence = ['P']
    tgt_sequence.extend(sentences[1].split()+sentences[2].split())
    tgt_list = []
    '''
    如果用list(set(word_sequence))来去重,得到的将是一个随机顺序的列表(因为set无序),
    这样得到的字典不同,保存的上一次训练的模型很有可能在这一次不能用
    (比如上一次的模型预测碰见i:0,love:1,就输出you:2,但这次模型you在字典3号位置,也就无法输出正确结果)
    '''
    for word in tgt_sequence:
        if word not in tgt_list:
            tgt_list.append(word)
    tgt_dict = {w:i for i,w in enumerate(tgt_list)}
    number_dict = {i:w for i,w in enumerate(tgt_dict)}
    tgt_dict_size = len(tgt_dict)
    tgt_len = len(sentences[1].split()) # length of target

    return src_dict,src_dict_size,tgt_dict,number_dict,tgt_dict_size,src_len,tgt_len

'''根据句子数据,构建词元的输入向量'''
def make_batch(sentences):
    input_batch = [[src_dict[n] for n in sentences[0].split()]]
    output_batch = [[tgt_dict[n] for n in sentences[1].split()]]
    target_batch = [[tgt_dict[n] for n in sentences[2].split()]]
    input_batch = torch.LongTensor(np.array(input_batch)).to(device)
    output_batch = torch.LongTensor(np.array(output_batch)).to(device)
    target_batch = torch.LongTensor(np.array(target_batch)).to(device)
    # print(input_batch, output_batch,target_batch) # tensor([[0, 1, 2, 3, 4, 5]]) tensor([[3, 1, 0, 2, 4]]) tensor([[1, 0, 2, 4, 5]])
    return input_batch, output_batch,target_batch

def get_position_encoding_table(n_position, d_model): 
    # inputs: (src_len+1, d_model) or (tgt_len+1, d_model)
    pos_table = np.zeros((n_position, d_model))
    for pos in range(n_position):
        for i in range(d_model):
            tmp = pos / np.power(10000, 2*(i//2) / d_model)
            if i % 2 == 0: 
                # 偶数为正弦
                pos_table[pos][i] = np.sin(tmp) # (7 or 6, 512)
            else:
                # 奇数为余弦
                pos_table[pos][i] = np.cos(tmp) # (7 or 6, 512)

    return torch.FloatTensor(pos_table).to(device)

def get_attn_pad_mask(seq_q, seq_k, dict): 
    '''mask大小和(len_q,len_k)一致,
    是为了在点积注意力中,与torch.matmul(Q,K)的大小一致'''
    # (seq_q, seq_k): (dec_inputs, enc_inputs) 
    # dec_inputs:[batch_size, tgt_len] # [1,5]
    # enc_inputs:[batch_size, src_len] # [1,6]
    batch_size, len_q = seq_q.size() # 1,5
    batch_size, len_k = seq_k.size() # 1,6
    """Tensor.data.eq(element)
    eq即equal,对Tensor中所有元素进行判断,和element相等即为True,否则为False,返回二值矩阵
    Examples:
        >>> tensor = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
        >>> tensor.data.eq(1) 
        tensor([[ True, False, False],
                [False, False, False]])
    """
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(dict['P']).unsqueeze(1) # 升维 enc: [1,6] -> [1,1,6]
    # 矩阵扩充: enc: pad_attn_mask: [1,1,6] -> [1,5,6]
    return pad_attn_mask.expand(batch_size, len_q, len_k) # batch_size, len_q, len_k

'''Attention = Softmax(Q * K^T) * V '''
def Scaled_Dot_Product_Attention(Q, K, V, attn_mask): 
    # Q_s: [batch_size, n_heads, len_q, d_k] # [1,8,5,64]
    # K_s: [batch_size, n_heads, len_k, d_k] # [1,8,6,64]
    # attn_mask: [batch_size, n_heads, len_q, len_k] # [1,8,5,6]

    """torch.matmul(Q, K)
        torch.matmul是tensor的乘法,输入可以是高维的.
        当输入是都是二维时,就是普通的矩阵乘法.
        当输入有多维时,把多出的一维作为batch提出来,其他部分做矩阵乘法.
        Exeamples:
            >>> a = torch.ones(3,4)
            >>> b = torch.ones(4,2)
            >>> torch.matmul(a,b).shape
            torch.Size([3,2])   

            >>> a = torch.ones(5,3,4)
            >>> b = torch.ones(4,2)
            >>> torch.matmul(a,b).shape
            torch.Size([5,3,2])

            >>> a = torch.ones(2,5,3)
            >>> b = torch.ones(1,3,4)
            >>> torch.matmul(a,b).shape
            torch.Size([2,5,4])
        """
    # [1,8,5,64] * [1,8,64,6] -> [1,8,5,6]
    # scores : [batch_size, n_heads, len_q, len_k]
    scores = torch.matmul(Q, K.transpose(2,3)) / np.sqrt(d_k) # divided by scale

    """scores.masked_fill_(attn_mask, -1e9) 
    由于scores和attn_mask维度相同,根据attn_mask中的元素值,把和attn_mask中值为True的元素的
    位置相同的scores元素的值赋为-1e9
    """
    scores.masked_fill_(attn_mask, -1e9)

    # 'P'的scores元素值为-1e9, softmax值即为0
    softmax = nn.Softmax(dim=-1) # 求行的softmax
    attn = softmax(scores) # [1,8,6,6]
    # [1,8,6,6] * [1,8,6,64] -> [1,8,6,64]
    context = torch.matmul(attn, V) # [1,8,6,64]
    return context, attn

class MultiHeadAttention(nn.Module):
    # dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
    def __init__(self):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_k*n_heads) # (512, 64*8) # d_q必等于d_k
        self.W_K = nn.Linear(d_model, d_k*n_heads) # (512, 64*8) # 保持维度不变
        self.W_V = nn.Linear(d_model, d_v*n_heads) # (512, 64*8)
        self.linear = nn.Linear(n_heads*d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        # dec_outputs: [batch_size, tgt_len, d_model] # [1,5,512]
        # enc_outputs: [batch_size, src_len, d_model] # [1,6,512]
        # dec_enc_attn_mask: [batch_size, tgt_len, src_len] # [1,5,6]
        # q/k/v: [batch_size, len_q/k/v, d_model]
        residual, batch_size = Q, len(Q)
        '''用n_heads=8把512拆成64*8,在不改变计算成本的前提下,让各注意力头相互独立,更有利于学习到不同的特征'''
        # Q_s: [batch_size, len_q, n_heads, d_q] # [1,5,8,64]
        # new_Q_s: [batch_size, n_heads, len_q, d_q] # [1,8,5,64]
        Q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  
        # K_s: [batch_size, n_heads, len_k, d_k] # [1,8,6,64]
        K_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  
        # V_s: [batch_size, n_heads, len_k, d_v] # [1,8,6,64]
        V_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  

        # attn_mask : [1,5,6] -> [1,1,5,6] -> [1,8,5,6]
        # attn_mask : [batch_size, n_heads, len_q, len_k]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) 

        # context: [batch_size, n_heads, len_q, d_v]
        # attn: [batch_size, n_heads, len_q(=len_k), len_k(=len_q)]
        # context: [1,8,5,64] attn: [1,8,5,6]
        context, attn = Scaled_Dot_Product_Attention(Q_s, K_s, V_s, attn_mask)
        """contiguous() 连续的
        contiguous: view只能用在连续(contiguous)的变量上.
        如果在view之前用了transpose, permute等,
        需要用contiguous()来返回一个contiguous copy
        """
        # context: [1,8,5,64] -> [1,5,512]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        # context: [1,5,512] -> [1,5,512]
        output = self.linear(context)
        """nn.LayerNorm(output) 样本归一化
        和对所有样本的某一特征进行归一化的BatchNorm不同,
        LayerNorm是对每个样本进行归一化,而不是一个特征

        Tips:
            归一化Normalization和Standardization标准化区别:
            Normalization(X[i]) = (X[i] - np.min(X)) / (np.max(X) - np.min(X))
            Standardization(X[i]) = (X[i] - np.mean(X)) / np.var(X)
        """
        output = self.layer_norm(output + residual)
        return output, attn 

class Position_wise_Feed_Forward_Networks(nn.Module):
    def __init__(self):
        super().__init__()
        '''输出层相当于1*1卷积层,也就是全连接层'''
        """nn.Conv1d
        in_channels应该理解为嵌入向量维度,out_channels才是卷积核的个数(厚度)
        """
        # 512 -> 2048
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        # 2048 -> 512
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        # enc_outputs: [batch_size, source_len, d_model] # [1,6,512]
        residual = inputs 
        relu = nn.ReLU()
        # output: 512 -> 2048 [1,2048,6]
        output = relu(self.conv1(inputs.transpose(1, 2)))
        # output: 2048 -> 512 [1,6,512]
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_attn = MultiHeadAttention()
        self.pos_ffn = Position_wise_Feed_Forward_Networks()

    def forward(self, enc_outputs, enc_attn_mask):
        # enc_attn_mask: [1,6,6]
        # enc_outputs to same Q,K,V
        # enc_outputs: [batch_size, source_len, d_model] # [1, 6, 512]
        enc_outputs, attn = self.enc_attn(enc_outputs, \
            enc_outputs, enc_outputs, enc_attn_mask) 
        # enc_outputs: [batch_size , len_q , d_model]
        enc_outputs = self.pos_ffn(enc_outputs) 
        return enc_outputs, attn

class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.dec_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = Position_wise_Feed_Forward_Networks()

    def forward(self, dec_outputs, enc_outputs, dec_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_attn = \
            self.dec_attn(dec_outputs, dec_outputs, dec_outputs, dec_attn_mask)
        # dec_outputs: [1, 5, 512]   dec_enc_attn: [1, 8, 5, 6]
        dec_outputs, dec_enc_attn = \
            self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        # 相当于两个全连接层 512 -> 2048 -> 512
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_attn, dec_enc_attn

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入向量的嵌入矩阵
        self.src_emb = nn.Embedding(src_dict_size, d_model).to(device)
        # 6个编码层
        self.layers = nn.ModuleList([EncoderLayer().to(device) for _ in range(n_layers)])

    def forward(self, enc_inputs): # enc_inputs: [batch_size, source_len] # [1, 6]
        input = [src_dict[i] for i in sentences[0].split()] # [0,1,2,3,4,5]
        
        '''加入pos_emb的意义: 如果不加,所有位置的单词都将有完全相同的影响,体现不出序列的特点'''
        # 可学习的输入向量嵌入矩阵 + 不可学习的序列向量矩阵
        # enc_outputs: [1, 6, 512]
        '''embbeding和linear不同,emb之后会加一个维度'''
        enc_outputs = self.src_emb(enc_inputs) + position_encoding[input] 
        # 屏蔽P,返回一个 [batch_size,src_len,src_len]=[1,6,6]的二值矩阵
        enc_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs, src_dict)

        enc_attns = []
        for layer in self.layers:
            # enc_outputs既是输入,也是输出
            enc_outputs, enc_attn = layer(enc_outputs, enc_attn_mask)
            enc_attns.append(enc_attn)
        return enc_outputs, enc_attns

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tgt_emb = nn.Embedding(tgt_dict_size, d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs): 
        # dec_inputs : [batch_size, target_len] [1,5]
        input = [tgt_dict[i] for i in sentences[1].split()]
        dec_outputs = self.tgt_emb(dec_inputs)

        # 为输入句子的每一个单词(不去重)加入序列信息
        # dec_outputs: [1, 5, 512]
        dec_outputs = dec_outputs + position_encoding[input] 
        
        # 屏蔽pad字符,返回一个 [batch_size,tgt_len,tgt_len]=[1,5,5]的二值矩阵
        dec_attn_mask = get_attn_pad_mask(dec_inputs, dec_inputs, tgt_dict)
        
        # 屏蔽pad字符和之后时刻的信息
        for i in range(0, len(dec_inputs[0])):
            for j in range(i+1, len(dec_inputs[0])):
                dec_attn_mask[0][i][j] = True # 使softmax值为0
        
        # 第二个多注意力机制,输入来自encoder和decoder 屏蔽encoder中的pad字符
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs, src_dict) # [1,5,6]

        dec_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_attn, dec_enc_attn = \
                layer(dec_outputs, enc_outputs, dec_attn_mask, dec_enc_attn_mask)
            dec_attns.append(dec_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_attns, dec_enc_attns

'''2.构建Transformer模型'''
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.projection = nn.Linear(d_model, tgt_dict_size, bias=False)
    def forward(self, dec_inputs, enc_inputs):
        enc_outputs, enc_attns = self.encoder(enc_inputs)
        dec_outputs, dec_attns, dec_enc_attns \
            = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        # model_outputs: [batch_size, tgt_len, tgt_dict_size]
        model_outputs = self.projection(dec_outputs) # [1,5,7]
        model_outputs = model_outputs.squeeze(0)

        return model_outputs, enc_attns, dec_attns, dec_enc_attns

'''贪婪搜索'''
def Greedy_Search(model, enc_inputs, start_symbol): # start_symbol = tgt_dict['S']
    '''
    依次生成tgt_len个目标单词,每生成一个单词都要进行一次完整的transformer计算,
    最后依次取第i个位置上概率最大的单词,生成新的dec_inputs
    贪婪搜索输出之后,新的dec_inputs重新和其他参数输入模型,最后生成predict
    '''
    padding_inputs = 'S' + ' P' * (tgt_len-1)
    # [1,5]
    dec_inputs = torch.Tensor([[tgt_dict[i] for i in padding_inputs.split()]]).to(device).type_as(enc_inputs.data)
    next_symbol = start_symbol
    i = 0

    while True:
        '''
        由enc_inputs和'S P P P P'生成i,然后i赋值给下一轮的dec_inputs:'S i P P P'
        然后生成want->'S i want P P' ······,生成beer->'S i want a beer'
        其实就是生成target_batch,然后错位赋值给dec_inputs,这个刚好符合训练模型时
        target_batch对应sentence[2],dec_inputs对应的sentence[1],enc_inputs对应的sentence[0]
        '''
        dec_inputs[0][i] = next_symbol
        outputs, _, _, _ = model(dec_inputs, enc_inputs)    
        predict = outputs.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_symbol = predict[i].item()
        i = i + 1
        # 超过最大长度或者有终止符则退出循环
        if next_symbol == tgt_dict['E'] or i > max_len:
            break
    outputs, _, _, _ = model(dec_inputs, enc_inputs)    
    predict = outputs.squeeze(0).max(dim=-1, keepdim=False)[1]
    return predict

# 为束搜索构造连续重复矩阵
def repeat(inputs,k=2):
    """连续重复
    Examples:
        >>> a = torch.tensor([[1,2,3],[4,5,6]])
        >>> print(repeat(a))
        [[1,2,3],
         [1,2,3],
         [4,5,6],
         [4,5,6]]
        >>> import torch
        >>> print(torch.repeat(a))
        [[1,2,3],
         [4,5,6],
         [1,2,3],
         [4,5,6]]        
    """
    tmp = torch.zeros(inputs.size(0)*k, inputs.size(1), inputs.size(2)).type_as(inputs.data)
    pos = 0
    while pos < len(inputs):
        tmp[pos*k] = inputs[pos]
        for i in range(1,k):
            tmp[pos*k+i] = inputs[pos]
        pos = pos + 1
    return tmp

'''束搜索'''
def Beam_Search(model, enc_inputs, start_symbol, k=2):
    padding_inputs = 'S' + (tgt_len-1)*' P'
    # dec_inputs:[1,5]
    dec_inputs = torch.Tensor([[tgt_dict[i] for i in padding_inputs.split()]]).to(device).type_as(enc_inputs.data)
    # all_dec_inputs用来存储每一种dec_inputs
    # all_dec_inputs:[1,1,5]
    all_dec_inputs = dec_inputs.unsqueeze(0)
    # 由于第一个单词被定为'S',所以其实每轮生成的实际结果数是2,4,8,16,16(最后一轮复制为32,但因为马上就结束循环,不会再分别赋值)
    for i in range(tgt_len):
        # 每一个单词,all_dec_inputs都要重复k次,存放新生成的k**i个结果
        # 注意,repeat函数实现的是连续重复,而不是toch.repeat那样的整体间断重复
        all_dec_inputs = repeat(all_dec_inputs,k) 
        
        # 对于第i个单词,以步长k遍历all_dec_inputs(即前一轮的所有dec_inputs)
        # 分别将其作为模型输入
        for j in range(0,len(all_dec_inputs),k):
            # print(j)
            dec_inputs = all_dec_inputs[j] 
            outputs, _, _, _ = model(dec_inputs, enc_inputs)
            # 排序,得到索引
            indices = outputs.sort(descending=True).indices # [5,7]
            # 提取第i个单词的第k个可能,赋值给all_dec_inputs的第j+pos个样本
            # (因为连续重复,所以all_dec_inputs中第j个dec_inputs生成的
            # 输出就分布在j~j+k个dec_inputs)
            for pos in range(k):
                if i < tgt_len - 1:
                    # i+1表示留给下一轮用,这和贪婪搜索中的思想一致
                    all_dec_inputs[j+pos][0][i+1] = indices[i][pos]
    print(all_dec_inputs)
    '''评判所有dec_inputs,选出输出概率最大的'''
    result = []
    for dec_inputs in all_dec_inputs:
        sum = 0
        outputs, _, _, _ = model(dec_inputs, enc_inputs) 
        indexs = outputs.squeeze(0).max(dim=-1, keepdim=False)[1]
        for i in range(tgt_len):
            # 计算各个样本输出的单词概率之和
            sum = sum + outputs.data[i][indexs[i]]
        result.append(sum.item())
    '''可能过拟合输出错误解'''
    max_index = result.index(max(result))
    predict = all_dec_inputs[max_index]
    return predict

if __name__ == '__main__':
    chars = 30 * '*'
    # sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']
    sentences = ['我 要 喝 啤 酒 P', 'S i want a beer', 'i want a beer E']
    device = ['cuda:0' if torch.cuda.is_available() else 'cpu'][0]
    d_model = 512  # Embedding Size 嵌入向量维度
    d_ff = 2048  # FeedForward dimension 
    d_k = d_v = 64  # dimension of K(=Q), V 
    n_layers = 6  # number of Encoder of Decoder Layer 编解码器层数
    n_heads = 8  # number of heads in Multi-Head Attention 多注意力机制头数
    max_len = 5

    '''1.数据预处理'''
    src_dict,src_dict_size,tgt_dict,number_dict,tgt_dict_size,src_len,tgt_len = pre_process(sentences)
    position_encoding = get_position_encoding_table(max(src_dict_size, tgt_dict_size), d_model)
    enc_inputs, dec_inputs, target_batch = make_batch(sentences)

    '''2.构建模型'''
    model = Transformer()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    # Adam的效果非常不好
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    if os.path.exists('model_param.pt') == True:
        # 加载模型参数到模型结构
        model.load_state_dict(torch.load('model_param.pt', map_location=device))

    '''3.训练'''
    print('{}\nTrain\n{}'.format('*'*30, '*'*30))
    loss_record = []
    for epoch in range(1000):
        optimizer.zero_grad()
        outputs, enc_attns, dec_attns, dec_enc_attns \
            = model(dec_inputs, enc_inputs)
        outputs = outputs.to(device)
        loss = criterion(outputs, target_batch.contiguous().view(-1))
        loss.backward()
        optimizer.step()

        if loss >= 0.01: # 连续30轮loss小于0.01则提前结束训练
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
    # 贪婪算法
    predict = Greedy_Search(model, enc_inputs, start_symbol=tgt_dict["S"])
    # 束搜索算法
    predict = Beam_Search(model, enc_inputs, start_symbol=tgt_dict["S"])
    
    for word in sentences[0].split():
        if word not in ['S', 'P', 'E']:
            print(word, end='')
    print(' ->',end=' ')
    
    for i in predict.data.squeeze():
        if number_dict[int(i)] not in ['S', 'P', 'E'] :
            print(number_dict[int(i)], end=' ')
