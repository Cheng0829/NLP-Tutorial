import os
import sys
import math
import pandas as pd
import numpy as np

# 读取文件
# tsv和csv文件的区别在它是用tab来分隔各个数据。
def read(fileName):
    tsv_file = pd.read_csv(fileName, sep = '\t', header=0, usecols=["SentenceId", "Phrase", "Sentiment"])
    return tsv_file


# 构建词库
def words_storage(fill_words_matrix): 
    fill_words_matrix = fill_words_matrix.reshape(-1,1) # 拉平
    # print(fill_words_matrix.shape)
    unique_words_matrix = np.unique(fill_words_matrix) # 去重
    sort_words_matrix = np.sort(unique_words_matrix)  # 排序
    sort_words_list = list(sort_words_matrix)
    # 词库字典
    words_dict = {sort_words_list.index(word):word for word in sort_words_list}
    # print('words_dict:',words_dict) # 词库
    return words_dict 

# 根据值查找字典键
def value_to_key(words_dict,value):
    for k,v in words_dict.items():
        if v == value:
            return k

# 分词,并构建二维词库矩阵
def divide_word(words_id_matrix):  
    m = words_id_matrix.shape[0] # 样本数
    words_list = []
    words_unique_list = []
    for i in range(m):  # 分词，装入不等长二维数组
        tmp = words_id_matrix[i].split(' ')
        words_list.append(tmp) # words_list是不等长二维数组
        words_unique_list.extend(tmp) # words_uique_list把所有单词放到一个列表中
    # 单词去重排序
    words_unique_list = sorted(list(set(words_unique_list)))
    words_count = len(words_unique_list) # 词库单词总数

    # 构建词典
    words_dict = {i:words_unique_list[i] for i in range(words_count)}

    # 记录单词出现次数
    words_count_matrix = np.zeros((m, words_count))
    for i in range(m):
        for j in range(len(words_list[i])):
            # 根据字典键值对,查找每个单词对应的序号
            value = words_list[i][j]
            key = value_to_key(words_dict,value)
            words_count_matrix[i][key] = words_count_matrix[i][key] + 1 
    # print(words_count_matrix)
    return words_count_matrix


def pre_process(fileName):  # 读取并预处理
    data = read(fileName)
    n, char = 20, '*'
    print('\n' + char * n + '数据读取完成!' + char * n + '\n')
    print('\n' + char * n + '数据预处理中!' + char * n+ '\n')
    words_id_matrix = np.array(data['Phrase'])     # 词句
    x_matrix = divide_word(words_id_matrix)
    # print("words_id_matrix:\n", words_id_matrix)
    # print("x_matrix:\n", x_matrix)
    # print(words_dict)
    y = np.array(data['Sentiment'])  # 情绪标签
    y = y.reshape(len(y),-1)
    print('\n' + char * n + '数据预处理完成!' + char * n + '\n')
    return x_matrix, y

pre_process('./try-train.tsv')
print()
# x矩阵横轴应该是单词总数,而不是最长句的单词个数
