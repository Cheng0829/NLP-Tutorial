# Transformer

[toc]

---

## 1.理论

### 1.1 Model Structure

![ ](img/Transformer-Model-Structure.png)

---

### 1.2 Multi-Head Attention & Scaled Dot-Product Attention

![ ](img/Transformer-Attention.png)

---

## 2.实验

### 2.1 束搜索

**束搜索过程示意图:**

![ ](img/Transformer-Beam-Search-one.png)
![ ](img/Transformer-Beam-Search-two.png)

---

### 2.2 Issue

1) **贪婪搜索和束搜索**
贪婪搜索和束搜索都是针对多个时间步,每一轮都要比较概率大小的,因此所有预测生成1个单词或者进行单词翻译的
都谈不上贪婪搜索和束搜索(没有多个时间步),直接用predict=model(inputs)的也谈不上贪婪搜索和束搜索(没有每一轮比较概率大小).
对于Seq2Seq和采用了序列模型的transformer来说,贪婪搜索和束搜索都应该用预测的单词覆盖填充的'SPPPP'中的'P'
   1) 对于翻译多个单词的任务,应该对于每个生成的单词设置循环
       1) 贪婪搜索:每个时间步生成一个单词的概率分布,取最大值,然后把这个值传给进行下一时间步,最后生成所有单词
       2) 束搜索(k=3):在每个时间步上预测k个max单词然后把这两个单词分别作为值传给进行下一时间步,
    当然,这样会进行$k^T$次预测,存储$k^T$个输出,最后取总概率最高的.
   2) 对于预测生成多个单词的任务,应该对于每个生成的单词设置循环
       1) 贪婪搜索:在输入时间步之后,每个输出时间步生成一个单词的概率分布,取最大值,
    然后把这个值传给进行下一时间步,最后生成所有单词
       2) 束搜索(k=3):在输入时间步之后,每个输出时间步上预测k个max单词,然后把这两个单词分别作为值
    传给进行下一时间步,当然,这样会进行$k^T$次预测,存储$k^T$个输出,最后取总概率最高的.

2) **为什么Seq2Seq和基于序列模型的transformer直接用predict=model(enc_inputs, dec_inputs='SPPPPP')的效果不好,其中transformer效果尤其差,而其他模型还不错?**
直接用predict=model(enc_inputs, dec_inputs='SPPPPP')既不算贪婪搜索也不算束搜索,因为这样只在最后才比较概率大小,而贪婪搜索和束搜索每轮都要计算
   1) RNN/LSTM等模型不需要'SPPPP'填充,因此不会受到空白信息影响,可以直接生成,
而Seq2Seq/transformer会受到'SPPPP'影响,所以效果不好.
   2) 其中,序列模型和RNN/LSTM类似,一个时间步对应一个单词,一个decoder时间步对应初始输入为'P',
上一次的时间步输出可以影响下一时间步生成,所以效果不好不坏
   3) 而transformer每次输入输出都是以整个句子为单位,所以不存在上一次的时间步输出可以影响下一时间步生成,所以效果尤其差,必须用循环依次生成

3) **为什么束搜索中有时候其他句子总体评价更高?**
    模型过于复杂,训练样本太少,导致过拟合.
