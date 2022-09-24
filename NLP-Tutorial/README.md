# NLP-Tutorial

<p align="center"><img width="100" src="img/pytorch.gif"/></p>

<p align="center">
  <a style="text-decoration:none" href="https://blog.csdn.net/louise_trender/category_12004083.html?spm=1001.2014.3001.5482">
    <img src="https://img.shields.io/badge/CSDN-Cheng0829-orange.svg?style=flat-square" alt="Store link" />
  </a>  

  <a style="text-decoration:none" href="https://i.cnblogs.com/posts?cateId=2215665">
    <img src="https://img.shields.io/badge/博客园-CJK's BLOG-blue.svg?style=flat-square" alt="Store link" />
  </a>
</p>

---

[toc]

---

## 项目实例

### 1.基础模型

- 1.[Logistic](./1.Logistic(%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB)/) - **文本情感二分类**
  
  - **Paper** - [A maximum entropy approach to natural language processing(1996)](https://aclanthology.org/J96-1002.pdf)
  - **Colab** - [Logistic.ipynb](https://**Colab**.research.google.com/github/Cheng0829/NLP/blob/master/nlp-tutorial/1.Logistic(%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB)/Logistic.ipynb)

- 2. [Word2Vec(Skip-gram)](./2.Word2Vec(生成文本嵌入矩阵)) - **生成文本嵌入矩阵并可视化**
  - **Paper** - [Distributed Representations of Words and Phrases
    and their Compositionality(2013)](https://**Paper**s.nips.cc/**Paper**/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
  - **Colab** - [Word2Vec.ipynb](https://**Colab**.research.google.com/github/Cheng0829/NLP/blob/master/nlp-tutorial/2.Word2Vec(%E7%94%9F%E6%88%90%E6%96%87%E6%9C%AC%E5%B5%8C%E5%85%A5%E7%9F%A9%E9%98%B5)/Word2Vec-Skipgram(Softmax).ipynb))

### 2.卷积神经网络

- 3. [TextCNN](./3.TextCNN(文本情感二分类)) - **文本情感二分类**
  - **Paper** - [Convolutional Neural Networks for Sentence Classification(2014)](http://www.aclweb.org/anthology/D14-1181)
  - **Colab** - [TextCNN.ipynb](https://**Colab**.research.google.com/github/Cheng0829/NLP/blob/master/nlp-tutorial/3.TextCNN(文本情感二分类)/TextCNN.ipynb)

### 3. 循环神经网络

- 4. [TextRNN](./4.TextRNN(预测下一个单词)) - **预测下一个单词**
  - **Paper** - [Finding Structure in Time(1990)](http://psych.colorado.edu/~kimlab/Elman1990.pdf)
  - **Colab** - [TextRNN.ipynb](https://**Colab**.research.google.com/github/Cheng0829/NLP/blob/master/nlp-tutorial/4.TextRNN(%E9%A2%84%E6%B5%8B%E4%B8%8B%E4%B8%80%E4%B8%AA%E5%8D%95%E8%AF%8D)/TextRNN.ipynb)

- 5. [TextLSTM](./5.TextLSTM(预测单词下一个字母)) - **预测单词下一个字母**
  - **Paper** - [LONG SHORT-TERM MEMORY(1997)](https://www.bioinf.jku.at/publications/older/2604.pdf)
  - **Colab** - [TextLSTM.ipynb](https://**Colab**.research.google.com/github/Cheng0829/NLP/blob/master/nlp-tutorial/5.TextLSTM(%E9%A2%84%E6%B5%8B%E5%8D%95%E8%AF%8D%E4%B8%8B%E4%B8%80%E4%B8%AA%E5%AD%97%E6%AF%8D)/TextLSTM.ipynb)

- 6. [Bi-LSTM](6.Bi-LSTM(在长句中预测下一个单词)) - **在长句中预测下一个单词**
  - **Paper** - [Bidirectional Long Short-Term Memory Networks for Relation Classification(2015)](https://aclanthology.org/Y15-1009.pdf)
  - **Colab** - [Bi_LSTM.ipynb](https://**Colab**.research.google.com/github/Cheng0829/NLP/blob/master/nlp-tutorial/6.Bi-LSTM(%E5%9C%A8%E9%95%BF%E5%8F%A5%E4%B8%AD%E9%A2%84%E6%B5%8B%E4%B8%8B%E4%B8%80%E4%B8%AA%E5%8D%95%E8%AF%8D)/Bi-LSTM.ipynb)

### 4. 序列模型

- 7. [Seq2Seq](7.Seq2Seq(单词翻译)) - **单词翻译**
  - **Paper** - [Learning Phrase Representations using RNN Encoder–Decoder
    for Statistical Machine Translation(2014)](https://arxiv.org/pdf/1406.1078.pdf)
  - **Colab** - [Seq2Seq.ipynb](https://**Colab**.research.google.com/github/Cheng0829/NLP/blob/master/nlp-tutorial/7.Seq2Seq(%E5%8D%95%E8%AF%8D%E7%BF%BB%E8%AF%91)/Seq2Seq.ipynb)

- 8. [Seq2Seq with Attention](4-2.Seq2Seq(Attention)) - **句子翻译**
  - **Paper** - [Neural Machine Translation by Jointly Learning to Align and Translate(2014)](https://arxiv.org/abs/1409.0473)
  - **Colab** - [Seq2Seq(Attention).ipynb]()

- 9. [Bi-LSTM with Attention](4-3.Bi-LSTM(Attention)) - **文本情感二分类**
  - **Paper** - [Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification(2016)](https://aclanthology.org/P16-2034.pdf)
  - **Colab** - [Bi_LSTM(Attention).ipynb]()

### 5. Model based on Transformer

- 10. [The Transformer](5-1.Transformer) - **机器翻译**
  - **Paper** - [Attention Is All You Need(2017)](https://arxiv.org/abs/1706.03762)
  - **Colab** - [Transformer.ipynb]()

- 11. [BERT](5-2.BERT) - **预测掩码标记和句间关系判断**
  - **Paper** - [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding(2018)](https://arxiv.org/abs/1810.04805)
  - **Colab** - [BERT.ipynb]()

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
