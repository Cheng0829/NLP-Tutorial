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

def h_like_y(h):
    new_h = np.zeros(h.shape)
    for i in range(h.shape[0]):
        max = np.argmax(h[i])
        new_h[i][max] = 1
    return new_h

def sigmoid(x):
    # print('sigmoid:',1/(1+np.exp(-x)))
    return 1/(1+np.exp(-x))


def hypothesis(theta, x_matrix):
    h = sigmoid(np.dot(theta.T, x_matrix.T)).T
    for i in range(x_matrix.shape[0]):
        h[i] = h[i] #/ sentence_length(x_matrix[i])
    return h


def cost_J(theta, x_matrix, lamda, m, y):
    h = hypothesis(theta, x_matrix)
    # print(h)
    #print((theta, x_matrix))
    return -1/m * np.sum(np.multiply(y, np.log(h)) +
                         np.multiply((1-y), np.log(1-h))) + \
        lamda * np.sum(np.multiply(theta, theta), axis=0)

def gradient(i, j, m, x_matrix, y, lamda, theta):
    h = hypothesis(theta, x_matrix)
    # h:(63,1) ; y:(63,5) ; h-y:(63,5)
    # x_matrix:(63,28) -> x:(1,28)
    return 1/m * np.sum((h - y) * x_matrix[i][j]) - np.sum(lamda/m * theta[j])


def gradient_descent(alpha, m, x_matrix, y, lamda, theta):
    h = hypothesis(theta, x_matrix)
    h = h_like_y(h)
    sum = np.sum((h - y),axis=0)
    for i in range(theta.shape[0]):
        for j in range(theta.shape[1]):
            theta[i][j] = theta[i][j] - alpha/m * (sum[j] * x_matrix[i][j] + lamda * theta[i][j])
    # print(theta)
    return theta


# 读取数据
# ["PhraseId","SentenceId","Phrase","Sentiment"]
x_matrix, y = data_read.pre_process('./try-train.tsv')
theta = np.random.rand(x_matrix.shape[1], 5)  # (28,5)  # 权重矩阵:随机初始化
alpha, m, lamda = 0.01, x_matrix.shape[0], 0.5
n, char = 20, '*'
# print(y)

for epoch in range(125):
    # break
    print('\n' + char * n + 'epoch=%d' % (epoch+1) + char * n + '\n')
    print('cost_J:', cost_J(theta, x_matrix, lamda, m, y))
    # print('gradient_%d:'%j, gradient(i, j, m, x_matrix, y, lamda, theta))
    theta = gradient_descent(alpha, m, x_matrix, y, lamda, theta)


"""predict"""


h = hypothesis(theta, x_matrix)

predict_result = []
accuracy = 0
for i in range(m):
    predict_max = np.argmax(h[i])
    real_max = np.argmax(y[i])
    predict_result.append(predict_max)
    if(predict_max == real_max):
        accuracy = accuracy + 1
real_result = []
for i in range(m):
    real_max = np.argmax(y[i])
    real_result.append(real_max)
print(real_result)
print(predict_result)
print('Accuracy:{}/{}={}'.format(accuracy,m,round(accuracy/m,2)))
