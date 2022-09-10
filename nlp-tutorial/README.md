# nlp-tutorial

<p align="center"><img width="100" src="https://media-thumbs.golden.com/OLqzmrmwAzY1P7Sl29k2T9WjJdM=/200x200/smart/golden-storage-production.s3.amazonaws.com/topic_images/e08914afa10a4179893eeb07cb5e4713.png" /></p>

`nlp-tutorial`基于
https://github.com/graykode/nlp-tutorial 中的代码,进行了改编,添加了大量中文注释和markdown笔记内容

## 项目实例

### 1.基本嵌入模型

- 1-1. [Word2Vec(Skip-gram)](1-2.Word2Vec) - **嵌入文字和显示图形**
  - Paper - [Distributed Representations of Words and Phrases
    and their Compositionality(2013)](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
  - Colab - [Word2Vec.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/1-2.Word2Vec/Word2Vec_Skipgram(Softmax).ipynb)
- 1-2. [FastText(Application Level)](1-3.FastText) - **文本分类**
  - Paper - [Bag of Tricks for Efficient Text Classification(2016)](https://arxiv.org/pdf/1607.01759.pdf)
  - Colab - [FastText.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/1-3.FastText/FastText.ipynb)

### 2.卷积神经网络

- 2-1. [TextCNN](2-1.TextCNN) - **情感二分类**
  - Paper - [Convolutional Neural Networks for Sentence Classification(2014)](http://www.aclweb.org/anthology/D14-1181)
  - [TextCNN.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/2-1.TextCNN/TextCNN.ipynb)

### 3. RNN(Recurrent Neural Network)

- 3-1. [TextRNN](3-1.TextRNN) - **预测下一个单词**
  - Paper - [Finding Structure in Time(1990)](http://psych.colorado.edu/~kimlab/Elman1990.pdf)
  - Colab - [TextRNN.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/3-1.TextRNN/TextRNN.ipynb)
- 3-2. [TextLSTM](https://github.com/graykode/nlp-tutorial/tree/master/3-2.TextLSTM) - **预测单词下一个字母**
  - Paper - [LONG SHORT-TERM MEMORY(1997)](https://www.bioinf.jku.at/publications/older/2604.pdf)
  - Colab - [TextLSTM.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/3-2.TextLSTM/TextLSTM.ipynb)
- 3-3. [Bi-LSTM](3-3.Bi-LSTM) - **在长句中预测下一个单词**
  - Colab - [Bi_LSTM.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/3-3.Bi-LSTM/Bi_LSTM.ipynb)

### 4.注意力机制

- 4-1. [Seq2Seq](4-1.Seq2Seq) - **更换为关联词**
  - Paper - [Learning Phrase Representations using RNN Encoder–Decoder
    for Statistical Machine Translation(2014)](https://arxiv.org/pdf/1406.1078.pdf)
  - Colab - [Seq2Seq.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/4-1.Seq2Seq/Seq2Seq.ipynb)
- 4-2. [Seq2Seq with Attention](4-2.Seq2Seq(Attention)) - **机器翻译**
  - Paper - [Neural Machine Translation by Jointly Learning to Align and Translate(2014)](https://arxiv.org/abs/1409.0473)
  - Colab - [Seq2Seq(Attention).ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/4-2.Seq2Seq(Attention)/Seq2Seq(Attention).ipynb)
- 4-3. [Bi-LSTM with Attention](4-3.Bi-LSTM(Attention)) - **情感二分类**
  - Colab - [Bi_LSTM(Attention).ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/4-3.Bi-LSTM(Attention)/Bi_LSTM(Attention).ipynb)

### 5.基于Transformer的模型

- 5-1.  [The Transformer](5-1.Transformer) - **机器翻译**
  - Paper - [Attention Is All You Need(2017)](https://arxiv.org/abs/1706.03762)
  - Colab - [Transformer.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/5-1.Transformer/Transformer.ipynb), [Transformer(Greedy_decoder).ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/5-1.Transformer/Transformer(Greedy_decoder).ipynb)
- 5-2. [BERT](5-2.BERT) - **分类下一句话和预测掩码标记**
  - Paper - [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding(2018)](https://arxiv.org/abs/1810.04805)
  - Colab - [BERT.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/5-2.BERT/BERT.ipynb)

## Dependencies

- Python 3.5+
- Pytorch 1.0.0+

## Author

- Cheng Junkai @github.com/Cheng0829
- Email : Chengjunkai829@gmail.com
  
