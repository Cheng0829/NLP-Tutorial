# nlp-tutorial

<p align="center"><img width="100" src="https://media-thumbs.golden.com/OLqzmrmwAzY1P7Sl29k2T9WjJdM=/200x200/smart/golden-storage-production.s3.amazonaws.com/topic_images/e08914afa10a4179893eeb07cb5e4713.png" /></p>

`nlp-tutorial`基于
https://github.com/graykode/nlp-tutorial 中的代码,进行了改编,添加了大量中文注释和markdown笔记内容

## 项目实例

### 1.基础模型

- 1.[Logistic](./1.Logistic(%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB)/) - **文本情感二分类**
  - Colab - [Logistic.ipynb](https://colab.research.google.com/github/Cheng0829/NLP/blob/master/nlp-tutorial/1.Logistic(%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB)/Logistic.ipynb)

- 2. [Word2Vec(Skip-gram)](./2.Word2Vec(生成文本嵌入矩阵)) - **嵌入文字和显示图形**
  - Paper - [Distributed Representations of Words and Phrases
    and their Compositionality(2013)](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
  - Colab - [Word2Vec.ipynb](https://colab.research.google.com/github/Cheng0829/NLP/blob/master/nlp-tutorial/2.Word2Vec(%E7%94%9F%E6%88%90%E6%96%87%E6%9C%AC%E5%B5%8C%E5%85%A5%E7%9F%A9%E9%98%B5)/Word2Vec-Skipgram(Softmax).ipynb))

### 2.卷积神经网络

- 3. [TextCNN](./3.TextCNN(文本情感二分类)) - **情感二分类**
  - Paper - [Convolutional Neural Networks for Sentence Classification(2014)](http://www.aclweb.org/anthology/D14-1181)
  - [TextCNN.ipynb](https://colab.research.google.com/github/Cheng0829/NLP/blob/master/nlp-tutorial/3.TextCNN(文本情感二分类)/TextCNN.ipynb)

### 3. 循环神经网络

- 4. [TextRNN](./4.TextRNN(预测下一个单词)) - **预测下一个单词**
  - Paper - [Finding Structure in Time(1990)](http://psych.colorado.edu/~kimlab/Elman1990.pdf)
  - Colab - [TextRNN.ipynb](https://colab.research.google.com/github/Cheng0829/NLP/blob/master/nlp-tutorial/4.TextRNN(%E9%A2%84%E6%B5%8B%E4%B8%8B%E4%B8%80%E4%B8%AA%E5%8D%95%E8%AF%8D)/TextRNN.ipynb)

- 5. [TextLSTM](./5.TextLSTM(预测单词下一个字母)) - **预测单词下一个字母**
  - Paper - [LONG SHORT-TERM MEMORY(1997)](https://www.bioinf.jku.at/publications/older/2604.pdf)
  - Colab - [TextLSTM.ipynb](https://colab.research.google.com/github/Cheng0829/NLP/blob/master/nlp-tutorial/5.TextLSTM(%E9%A2%84%E6%B5%8B%E5%8D%95%E8%AF%8D%E4%B8%8B%E4%B8%80%E4%B8%AA%E5%AD%97%E6%AF%8D)/TextLSTM.ipynb)

- 6. [Bi-LSTM](6.Bi-LSTM(在长句中预测下一个单词)) - **在长句中预测下一个单词**
  - Colab - [Bi_LSTM.ipynb](https://colab.research.google.com/github/Cheng0829/NLP/blob/master/nlp-tutorial/6.Bi-LSTM(%E5%9C%A8%E9%95%BF%E5%8F%A5%E4%B8%AD%E9%A2%84%E6%B5%8B%E4%B8%8B%E4%B8%80%E4%B8%AA%E5%8D%95%E8%AF%8D)/Bi-LSTM.ipynb)

## Dependencies

- Python 3.5+
- Pytorch 1.0.0+

## Author

- Cheng Junkai @github.com/Cheng0829
- Email : Chengjunkai829@gmail.com
