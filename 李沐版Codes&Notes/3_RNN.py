# RNN
'''
import numpy as np
import torch,time,math
from torch import nn, optim
import torch.nn.functional as F
import sys
sys.path.append("..")
import mymodule
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化化模型参数
def get_params():
    def _one(shape):
        # np.random.normal(均值,标准差,输出值的维度)从正态分布中抽取随机样本。
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape),device=device, dtype=torch.float32)
        return torch.nn.Parameter(ts, requires_grad=True) #pytorch专用类型转换函数(生成参数属性,方便训练)
    # 隐藏层参数
    W_xh = _one((num_inputs, num_hiddens))
    # print(W_xh.shape)
    W_hh = _one((num_hiddens, num_hiddens))
    b_h = torch.nn.Parameter(torch.zeros(num_hiddens,device=device, requires_grad=True))
    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs,device=device, requires_grad=True))
    return nn.ParameterList([W_xh, W_hh, b_h, W_hq, b_q])

# BATCH_SIZE:即一次训练所抓取的数据样本数量
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )

# rnn函数定义了在⼀个时间步里如何计算隐藏状态和输出.这⾥的激活函数使用了tanh函数
# 根据前一个字符的onehot向量inputs,计算每一层的输入输出
def rnn(inputs, state, params):
    # inputs和outputs皆为num_steps个形状为(batch_size, vocab_size)的矩阵
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = torch.tanh(torch.matmul(X, W_xh) + torch.matmul(H,W_hh) + b_h)
        Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs,(H,)

# 加载歌词:
# char_to_idx是每一个字加上去重后的位置序号的字典
# idx_to_char是每一个字的列表
# vocab_size是总字数
# corpus_indices是一个列表,列表里的每个元素是与数据集中每个字符相对应的id(去重后的位置序号)
(corpus_indices, char_to_idx, idx_to_char, vocab_size) = mymodule.load_data_jay_lyrics()

# 隐藏单元个数num_hiddens是⼀个超参数
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size

num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 35, 32, 1e2, 1e-2

pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                          vocab_size, device, corpus_indices, idx_to_char,
                          char_to_idx, True, num_epochs, num_steps,
                          lr, clipping_theta, batch_size, pred_period,
                          pred_len, prefixes)
'''

# GRU
'''
# 读取数据
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import sys
sys.path.append("..")
import mymodule
from mymodule import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载歌词
(corpus_indices, char_to_idx, idx_to_char, vocab_size) = mymodule.load_data_jay_lyrics()

num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
print('will use', device)
def get_params():
    def _one(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape),device=device, dtype=torch.float32)
        return torch.nn.Parameter(ts, requires_grad=True)
    def _three():
        return (_one((num_inputs, num_hiddens)),_one((num_hiddens, num_hiddens)),torch.nn.Parameter(torch.zeros(num_hiddens,device=device, dtype=torch.float32), requires_grad=True))

    W_xz, W_hz, b_z = _three() # 更新门参数
    W_xr, W_hr, b_r = _three() # 重置门参数
    W_xh, W_hh, b_h = _three() # 候选隐藏状态参数

    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs,device=device, dtype=torch.float32), requires_grad=True)

    return nn.ParameterList([W_xz, W_hz, b_z, W_xr, W_hr, b_r,W_xh, W_hh, b_h, W_hq, b_q])

def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )

def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid(torch.matmul(X, W_xz) + torch.matmul(H,W_hz) + b_z)
        R = torch.sigmoid(torch.matmul(X, W_xr) + torch.matmul(H,W_hr) + b_r)
        H_tilda = torch.tanh(torch.matmul(X, W_xh) + R * torch.matmul(H, W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)

num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2

pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']

train_and_predict_rnn(gru, get_params, init_gru_state, num_hiddens,
                          vocab_size, device, corpus_indices, idx_to_char,
                          char_to_idx, True, num_epochs, num_steps,
                          lr, clipping_theta, batch_size, pred_period,
                          pred_len, prefixes)
'''

# LSTM
'''
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import sys,mymodule
from mymodule import *
sys.path.append("..")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
(corpus_indices, char_to_idx, idx_to_char, vocab_size) =load_data_jay_lyrics()
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
print('will use', device)

def get_params():
    def _one(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape),device=device, dtype=torch.float32)
        return torch.nn.Parameter(ts, requires_grad=True)
    def _three():
        return (_one((num_inputs, num_hiddens)),_one((num_hiddens, num_hiddens)),
                torch.nn.Parameter(torch.zeros(num_hiddens,device=device, dtype=torch.float32), requires_grad=True))

    W_xi, W_hi, b_i = _three()  # 输⼊门参数
    W_xf, W_hf, b_f = _three()  # 遗忘门参数
    W_xo, W_ho, b_o = _three()  # 输出门参数
    W_xc, W_hc, b_c = _three()  # 候选记忆细胞参数
    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs,device=device, dtype=torch.float32), requires_grad=True)
    return nn.ParameterList([W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q])

def init_lstm_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))

def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        I = torch.sigmoid(torch.matmul(X, W_xi) + torch.matmul(H, W_hi) + b_i)
        F = torch.sigmoid(torch.matmul(X, W_xf) + torch.matmul(H, W_hf) + b_f)
        O = torch.sigmoid(torch.matmul(X, W_xo) + torch.matmul(H, W_ho) + b_o)
        C_tilda = torch.tanh(torch.matmul(X, W_xc) + torch.matmul(H, W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * C.tanh()
        Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H, C)


num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2

pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']

train_and_predict_rnn(lstm, get_params, init_lstm_state, num_hiddens,
                      vocab_size, device, corpus_indices, idx_to_char,
                      char_to_idx, True, num_epochs, num_steps,
                      lr, clipping_theta, batch_size, pred_period,
                      pred_len, prefixes)
'''