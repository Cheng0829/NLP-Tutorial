import os
import sys
import math
import pandas as pd
import numpy as np
import data_read
from math import exp, log


# 返回填充矩阵中每个句子的实际长度
def sentence_length(sentence):
    return sum(sentence)

def sigmoid(x):
    # print('sigmoid:',1/(1+np.exp(-x)))
    return 1/(1+np.exp(-x))


def hypothesis(theta, x_matrix):
    h = sigmoid(np.dot(x_matrix, theta))
    for i in range(x_matrix.shape[0]):
        h[i] = h[i] # / sentence_length(x_matrix[i])
    return h


def cost_J(theta, x_matrix, lamda, m, y):
    h = hypothesis(theta, x_matrix)
    # print(h)
    #print((theta, x_matrix))
    return -1/m * (np.dot(y.T, np.log(h)) + np.dot((1-y).T, np.log(1-h))) - \
        lamda/m * np.dot(theta.T, theta)

def gradient(i, j, m, x_matrix, y, lamda, theta):
    h = hypothesis(theta, x_matrix)
    # h:(63,1) ; y:(63,1) ; h-y:(63,1)
    # x_matrix:(63,28) -> x:(1,28)
    return 1/m * np.sum((h - y) * x_matrix[i][j]) - np.sum(lamda/m * theta[j])


def gradient_descent(alpha, m, x_matrix, y, lamda, theta):
    h = hypothesis(theta, x_matrix)
    # print('x:', x_matrix.shape) #(63,28)
    # print('y:', y.shape) # (63,1)
    # print('h:', h.shape) # (63,1)
    # print('θ:', theta.shape) # (28,1)
    for i in range(x_matrix.shape[0]): # 对于每个句子 
        for j in range(x_matrix.shape[1]): # 
            # theta[j] = theta[j] - alpha/m * (np.sum(h-y) * x_matrix[i][j] + lamda * theta[j])
            theta[j] = theta[j] - alpha * np.sum(h-y) * x_matrix[i][j]
    # print(theta)
    return theta


# 读取数据
# ["PhraseId","SentenceId","Phrase","Sentiment"]
x_matrix, y = data_read.pre_process('./try-train.tsv')
theta = np.random.rand(x_matrix.shape[1], 1)  # (28,1)  # 权重矩阵:随机初始化
alpha, m, lamda = 0.1, x_matrix.shape[0], 0
n, char = 20, '*'
# print(y)

for epoch in range(55):
    # break
    print('\n' + char * n + 'epoch=%d' % (epoch+1) + char * n + '')
    print('cost_J:', cost_J(theta, x_matrix, lamda, m, y))
    # print('gradient_%d:'%j, gradient(i, j, m, x_matrix, y, lamda, theta))
    theta = gradient_descent(alpha, m, x_matrix, y, lamda, theta)
    #break


"""predict"""

zero = np.zeros(theta.shape)
h = hypothesis(zero, x_matrix)
h = hypothesis(theta, x_matrix)
# print(theta)
print(h)
predict_result, real_result = [], []
accuracy = 0
for i in range(m):
    yi = y.T[0][i] 
    real_result.append(int(yi))
    if(h[i]>=0.5):
        result = 1
    else:
        result = 0
    predict_result.append(result)
    if(yi==result):
        accuracy = accuracy + 1
print(real_result)
print(predict_result)
print('Accuracy:{}/{}={}'.format(accuracy,m,round(accuracy/m,2)))
