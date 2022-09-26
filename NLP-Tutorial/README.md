# NLP-Tutorial

---

<p align="center"><img width="150" src="img/pytorch.gif"/></p>

<p align="center">
  <a style="text-decoration:none"  href="https://blog.csdn.net/louise_trender/category_12004083.html?spm=1001.2014.3001.5482">
    <img src="https://img.shields.io/badge/CSDN-Cheng0829-orange.svg?style=flat-square" width="150" alt="Store link" />
  </a>  

  <a style="text-decoration:none" href="https://i.cnblogs.com/posts?cateId=2215665">
    <img src="https://img.shields.io/badge/博客园-CJK's BLOG-blue.svg?style=flat-square" width="150" alt="Store link" />
  </a>
</p>

---

[toc]

---

## 项目实例

### 1.基础模型

- 1.[Logistic](./1.Logistic(%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB)/) - **文本情感二分类**
  
  - **Paper** - [A maximum entropy approach to natural language processing(1996)](https://aclanthology.org/J96-1002.pdf)
  - **Colab** - [Logistic.ipynb](https://colab.research.google.com/github/Cheng0829/NLP/blob/master/NLP-Tutorial/1.Logistic(文本分类)/Logistic.ipynb)

- 2. [Word2Vec(Skip-gram)](./2.Word2Vec(生成文本嵌入矩阵)) - **生成文本嵌入矩阵并可视化**
  - **Paper** - [Distributed Representations of Words and Phrases
    and their Compositionality(2013)](https://**Paper**s.nips.cc/**Paper**/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
  - **Colab** - [Word2Vec.ipynb](https://colab.research.google.com/github/Cheng0829/NLP/blob/master/NLP-Tutorial/2.Word2Vec(生成文本嵌入矩阵)/Word2Vec.ipynb)

### 2.卷积神经网络

- 3. [TextCNN](./3.TextCNN(文本情感二分类)) - **文本情感二分类**
  - **Paper** - [Convolutional Neural Networks for Sentence Classification(2014)](http://www.aclweb.org/anthology/D14-1181)
  - **Colab** - [TextCNN.ipynb](https://colab.research.google.com/github/Cheng0829/NLP/blob/master/NLP-Tutorial/3.TextCNN(文本情感二分类)/TextCNN.ipynb)

### 3. 循环神经网络

- 4. [TextRNN](./4.TextRNN(预测下一个单词)) - **预测下一个单词**
  - **Paper** - [Finding Structure in Time(1990)](http://psych.colorado.edu/~kimlab/Elman1990.pdf)
  - **Colab** - [TextRNN.ipynb](https://colab.research.google.com/github/Cheng0829/NLP/blob/master/NLP-Tutorial/4.TextRNN(预测下一个单词)/TextRNN.ipynb)

- 5. [TextLSTM](./5.TextLSTM(预测单词下一个字母)) - **预测单词下一个字母**
  - **Paper** - [LONG SHORT-TERM MEMORY(1997)](https://www.bioinf.jku.at/publications/older/2604.pdf)
  - **Colab** - [TextLSTM.ipynb](https://colab.research.google.com/github/Cheng0829/NLP/blob/master/NLP-Tutorial/5.TextLSTM(预测单词下一个字母)/TextLSTM.ipynb)

- 6. [Bi-LSTM](6.Bi-LSTM(在长句中预测下一个单词)) - **在长句中预测下一个单词**
  - **Paper** - [Bidirectional Long Short-Term Memory Networks for Relation Classification(2015)](https://aclanthology.org/Y15-1009.pdf)
  - **Colab** - [Bi_LSTM.ipynb](https://colab.research.google.com/github/Cheng0829/NLP/blob/master/NLP-Tutorial/6.Bi-LSTM(在长句中预测下一个单词))

### 4. 序列模型

- 7. [Seq2Seq](7.Seq2Seq(单词翻译)) - **单词翻译**
  - **Paper** - [Learning Phrase Representations using RNN Encoder–Decoder
    for Statistical Machine Translation(2014)](https://arxiv.org/pdf/1406.1078.pdf)
  - **Colab** - [Seq2Seq.ipynb](https://colab.research.google.com/github/Cheng0829/NLP/blob/master/NLP-Tutorial/7.Seq2Seq(单词翻译)/Seq2Seq.ipynb)

- 8. [Seq2Seq with Attention](4-2.Seq2Seq(Attention)) - **句子翻译**
  - **Paper** - [Neural Machine Translation by Jointly Learning to Align and Translate(2014)](https://arxiv.org/abs/1409.0473)
  - **Colab** - [Seq2Seq(Attention).ipynb](https://colab.research.google.com/github/Cheng0829/NLP/blob/master/NLP-Tutorial/8.Seq2Seq(Attention)(句子翻译)/Seq2Seq(Attention).ipynb)

- 9. [Bi-LSTM with Attention](4-3.Bi-LSTM(Attention)) - **文本情感二分类**
  - **Paper** - [Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification(2016)](https://aclanthology.org/P16-2034.pdf)
  - **Colab** - [Bi_LSTM(Attention).ipynb](https://colab.research.google.com/github/Cheng0829/NLP/blob/master/NLP-Tutorial/9.Bi-LSTM(Attention)(文本情感二分类)/Bi-LSTM(Attention).ipynb)

### 5. Model based on Transformer

- 10. [The Transformer](5-1.Transformer) - **机器翻译**
  - **Paper** - [Attention Is All You Need(2017)](https://arxiv.org/abs/1706.03762)
  - **Colab** - [Transformer.ipynb](https://colab.research.google.com/github/Cheng0829/NLP/blob/master/NLP-Tutorial/10.Transformer(机器翻译)/Transformer.ipynb)

- 11. [BERT](5-2.BERT) - **预测掩码标记和句间关系判断**
  - **Paper** - [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding(2018)](https://arxiv.org/abs/1810.04805)
  - **Colab** - [BERT.ipynb](https://colab.research.google.com/github/Cheng0829/NLP/blob/master/NLP-Tutorial/11.BERT(预测掩码标记和句间关系判断)/BERT.ipynb)

---

## Dependencies

- Python 3.6+
- Pytorch 1.0.0+
- numpy 1.10.0+

---

## Author

- **Cheng Junkai** @github.com/Cheng0829
- **Email :** Chengjunkai829@gmail.com
- **Site :** [CSDN](https://blog.csdn.net/Louise_Trender?type=blog) & [博客园](https://www.cnblogs.com/chengjunkai/)

---

## Reference

- 本项目基于[NLP-Tutorial](https://github.com/graykode/nlp-tutorial)进行了改写,并添加了大量中文注释和markdown笔记内容
