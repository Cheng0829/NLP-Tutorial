import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

# 读取数据，并删除不相关字段
# df_train = pd.read_csv('./train_set.csv')  #训练集
# df_test = pd.read_csv('./test_set.csv')   #测试集
df_train = pd.read_csv('./try-train.tsv', sep='\t')
df_test = pd.read_csv('./try-test.tsv', sep='\t')
df_train.drop(columns=['PhraseId', 'SentenceId'], inplace=True)
df_test.drop(columns=['PhraseId', 'SentenceId'], inplace=True)  #保留id
print(df_train)
"""
    @ngram_range:词组切分的长度范围
    @min_df：参数为int，小于这个值则认为不是关键词
    @max_df：参数是float，则表示词出现的次数与语料库文档数的最大百分比
    @max_features：默认为None，可设为int，对所有关键词的term frequency进行降序排序，只取前max_features个作为关键词集
"""
#将文本中的词语转换为词频矩阵
vectorizer = CountVectorizer(ngram_range=(1,2), min_df=3, max_df=0.9, max_features=100)
print(vectorizer)
#拟合生成标准化词频矩阵
x_train = vectorizer.fit_transform(df_train['Phrase'])
x_test = vectorizer.transform(df_test['Phrase'])
y_train = df_train['Sentiment'] - 1 #类别默认是从0开始的，所以这里需要-1

#构建逻辑回归模型
lg = LogisticRegression(C=5, dual=True)
lg.fit(x_train, y_train)

#预测
y_test = lg.predict(x_test)

#保存结果
df_test['Sentiment'] = y_test.tolist() #转化为矩阵列表形式存储
df_test['Sentiment'] =+ 1 #还原类别+1
df_result = df_test.loc[:, ['Phrase', 'Sentiment']] #根据index索引所有行和id,class列
df_result.to_csv('./result.csv', index=False)
'''
把数据集进行脱敏处理
'''