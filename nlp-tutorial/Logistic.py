"""
Task: 基于逻辑斯蒂回归的文本分类
Author: ChengJunkai @github.com/Cheng0829
Email: chengjunkai829@gmail.com
Date: 2022/09/03
Reference: Tae Hwan Jung(Jeff Jung) @graykode
"""

'''
数据集说明:
train.csv结构:
id	article	                 word_seg	        class
0   991700 509221 410256···  816903 597526···   14
test.csv结构:
id	article	                 word_seg	        
0   111111 222222 333333···  444444 555555···   
文章索引“id”、中文文字索引“article”、词语索引“word_seg”、文章类别“class”。
对于class,测试集是不包含该字段的,这就是我们需要分类的类别,也就是模型的输出y。
数据集都经过了脱敏处理,也就是说我们打开数据集文件看到的都是一些数字,
这些数字其实代表的就是一个字或词或标点符号。
本实验只从词的角度,只分析“word_seg”作为输入特征,“class”作为分类标记进行训练。不使用article字段
'''
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import time

# 读取csv文件(前n行)
def read(train_set_path, test_set_path):
    n = 100 # 只读前100行样本
    # 法1:选择性读取
    train_data = pd.read_csv(train_set_path, header=0, usecols=['word_seg', 'class'], nrows=n)  # 训练集
    test_data = pd.read_csv(test_set_path, header=0, usecols=['id', 'word_seg'], nrows=n) # 测试集
    # 法2:读取数据,并删除不相关字段
    # train_data = pd.read_csv(train_set_path, nrows=n)  #训练集
    # test_data = pd.read_csv(test_set_path, nrows=n)   #测试集
    # train_data.drop(columns=['id','article'], inplace=True)
    # test_data.drop(columns=['article'], inplace=True)  #保留id
    return train_data, test_data

# 训练拟合模型
def train(train_data):
    # 将文本中的词语转换为词频矩阵
    '''
    CountVectorizer是属于常见的特征数值计算类,是一个文本特征提取方法。
    对于每一个训练文本,它只考虑每种词汇在该训练文本中出现的频率。
    CountVectorizer会将文本中的词语转换为词频矩阵,它通过fit_transform函数计算各个词语出现的次数。
    '''
    vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, max_features=100000)
    '''
    ngram_range:词组切分的长度范围,ngram_range(1,2)是指将text根据N-gram规则,分别对每个词/每2个词进行划分
    min_df:单词出现次数小于这个值,则认为不是关键词(占比小,影响度小)
    max_df:表示有某单词出现的文档数与语料库文档数的最大百分比(太常见则没有区分度)
    stop_words:设置停用词
    # TF-IDF -> max_df, min_df
    max_features:默认为None,可设为int,对所有关键词的词频进行降序排序,只取前max_features个作为关键词集
    '''

    # CountVectorizer会将文本中的词语转换为词频矩阵,它通过fit_transform函数计算各个词语出现的次数。
    # 拟合生成标准化词频矩阵
    """
    fit_transform() = fit() + transform()
    fit():求得训练集X的均值,方差,最大值,最小值等属性 -> 和logistic.fit()不同
    transform():在fit的基础上,进行标准化,降维,归一化等操作
    Tips:正确做法:先对训练集用fit_transform进行fit,找到均值方差并标准化,然后在测试集中只transform,这样可以保证训练集和测试集的处理方式一样
    """
    # fit_transform()的作用就是先拟合数据,然后转化它将其转化为标准形式。相当于fit+transform
    x_train = vectorizer.fit_transform(train_data['word_seg'])
    """fit_transform/transform返回值的结构
    (sample_i,number_j)  count
    1.sample_i是第i个样本的序号,
    2.number_j是该单词/词组在整个数据集单词/词组中的序号
    (对于不同样本文档中相同的单词/词组,它们的number_j相同,但不同的单词/词组的序号不重复)
    在计算x_train.shape时,只会计算不同的单词/词组
    3.count是该单词/词组在该样本文档的频数
    """
    y_train = train_data['class'] - 1  # 类别默认是从0开始的,所以这里需要-1
    
    # 构建逻辑回归模型
    logistic_model = LogisticRegression(C=4, dual=False) 
    """LogisticRegression
    C:正浮点型.正则化强度的倒数(默认为1.0)
    dual = True则用对偶形式;dual = False则用原始形式(默认为False)
    """
    # 拟合,训练
    logistic_model.fit(x_train, y_train) 
    """
    LogisticRegression.fit(x,y):根据给定的训练数据对模型进行拟合。
    输入为样本的词频矩阵和样本类别
    """
    return vectorizer, logistic_model


# 保存预测结果
def save(test_data, y_test, save_path):
    # 保存结果
    y_test = y_test.tolist()  # 把(n,)的一维矩阵y_test转化为矩阵列表形式存储
    test_data['class'] = y_test + 1  # 新建一个class列,存入类别值(类别还原+1)
    data_result = test_data.loc[:, ['id', 'class']]  # 根据index索引所有行和id,class列
    """
    pandas的loc函数:locate定位函数.
    1.定位行列:data.loc[行索引, 列名]
    2.只定位行:data.loc[行索引]
    3.只定位列:data.loc[:, 列名]
    """
    data_result.to_csv(save_path, index=False)


if __name__=="__main__":
    chars = '*' * 20
    '''1.读取数据'''
    train_data, test_data = read(r'train_set.csv', r'test_set.csv')

    '''2.训练'''
    vectorizer, logistic_model = train(train_data)
    
    '''3.预测'''
    x_test = vectorizer.transform(test_data['word_seg']) # 矩阵标准化
    y_test = logistic_model.predict(x_test) # (n,)
    """
    LogisticRegression.predict(x,y):根据训练模型对测试集进行预测。
    输入为测试样本的词频矩阵,输出为(n,)的一维矩阵
    """

    '''4.保存'''
    # save(test_data, y_test,save_path='./result.csv')

    print('Over!')
    