import json
import multiprocessing
import os
import torch
from torch import nn
import d2l.pytorch as d2l

"""
***************************************************************************
SNLI数据集的下游任务自然语言推断
***************************************************************************
"""
# **Tips:** 有cuda的话可能需要修改d2l.torch(我把名字改为了d2l.pytorch,避免重名)中的sequence_mask函数,
# 在使用valid_len之前加一句`valid_len = valid_len.float() # cjk: 原来没有这一句`

"""加载预训练的BERT模型============================================================================="""
# 加载预先训练好的BERT参数
# 预训练好的BERT模型包含一个定义词表的vocab.json文件和一个预训练参数的pretrained.params文件

# BERT-微调.py通过json文件获取词汇数据,然后加载模型参数(虽然后缀为参数文件为"XXX.params",但和pt文件操作一样)
def load_pretrained_model(pretrained_model, num_hiddens, ffn_num_hiddens,
                          num_heads, num_layers, dropout, max_len, devices):
    data_dir = f'./data/{pretrained_model}.torch'
    # 定义空词表以加载预定义词表
    vocab = d2l.Vocab()
    # 这里通过json文件获取词汇数据
    vocab.idx_to_token = json.load(open(os.path.join(data_dir, 'vocab.json')))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(vocab.idx_to_token)}
    bert = d2l.BERTModel(len(vocab), num_hiddens, norm_shape=[256],
                         ffn_num_input=256, ffn_num_hiddens=ffn_num_hiddens,
                         num_heads=4, num_layers=2, dropout=0.2,
                         max_len=max_len, key_size=256, query_size=256,
                         value_size=256, hid_in_features=256,
                         mlm_in_features=256, nsp_in_features=256)
    # 加载预训练BERT参数
    # fine-tuning-parameter.pt
    bert.load_state_dict(torch.load(os.path.join(data_dir, 'pretrained.params')))
    # bert = torch.load('pretrained.pt')
    return bert, vocab


"""提取SNLI数据转换成BERT输入模式============================================================================="""
# 在每个样本中,前提和假设形成一对文本序列,并被打包成一个BERT输入序列
class SNLIBERTDataset(torch.utils.data.Dataset): # 定制的数据集类
    def __init__(self, dataset, max_len, vocab=None):
        # premise:前提 hypothesis:假设
        all_premise_hypothesis_tokens = [[p_tokens, h_tokens] for p_tokens, h_tokens 
            in zip(*[d2l.tokenize([s.lower() for s in sentences]) 
                for sentences in dataset[:2]])] # 分词
            
        self.labels = torch.tensor(dataset[2]) # return XX, YY, labels
        self.vocab = vocab
        self.max_len = max_len
        (self.all_token_ids, self.all_segments, 
            self.valid_lens) = self._preprocess(all_premise_hypothesis_tokens)
        print('read ' + str(len(self.all_token_ids)) + ' examples')

    # 预处理
    def _preprocess(self, all_premise_hypothesis_tokens): 
        pool = multiprocessing.Pool(4)  # 使用4个进程并行生成
        out = pool.map(self._mp_worker, all_premise_hypothesis_tokens) # 并行加速,分词填充
        all_token_ids = [token_ids for token_ids, segments, valid_len in out]
        all_segments = [segments for token_ids, segments, valid_len in out]
        valid_lens = [valid_len for token_ids, segments, valid_len in out] 
        return (torch.tensor(all_token_ids, dtype=torch.long),
                torch.tensor(all_segments, dtype=torch.long),
                torch.tensor(valid_lens))

    # 分词,分句,填充
    def _mp_worker(self, premise_hypothesis_tokens):
        p_tokens, h_tokens = premise_hypothesis_tokens
        self._truncate_pair_of_tokens(p_tokens, h_tokens) # 截断文本标记
        tokens, segments = d2l.get_tokens_and_segments(p_tokens, h_tokens) # 分词,分句
        token_ids = self.vocab[tokens] + [self.vocab['<pad>']] * (self.max_len - len(tokens))
        segments = segments + [0] * (self.max_len - len(segments))
        valid_len = len(tokens)
        return token_ids, segments, valid_len

    # 截断文本标记至max_len
    def _truncate_pair_of_tokens(self, p_tokens, h_tokens): 
        # 为BERT输入中的'<CLS>'、'<SEP>'和'<SEP>'词元保留位置
        while len(p_tokens) + len(h_tokens) > self.max_len - 3:
            if len(p_tokens) > len(h_tokens):
                p_tokens.pop()
            else:
                h_tokens.pop()

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],self.valid_lens[idx]), self.labels[idx]
                 
    def __len__(self):
        return len(self.all_token_ids)
        

"""在BERT输出之上再加一个MLP============================================================================="""
# 只需要一个额外的多层感知机，该多层感知机由两个全连接层组成(self.hidden和self.output),
# 这个多层感知机将特殊的"<cls>"词元的BERT表示进行了转换,
class BERTClassifier(nn.Module):
    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        self.encoder = bert.encoder
        self.hidden = bert.hidden
        self.output = nn.Linear(256, 3) # 自然语言推断的三个输出：蕴涵、矛盾和中性.
    def forward(self, inputs):
        tokens_X, segments_X, valid_lens_x = inputs
        encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x)
        return self.output(self.hidden(encoded_X[:, 0, :]))

if __name__ == '__main__':
    devices = d2l.try_all_gpus()
    bert, vocab = load_pretrained_model('bert.small', num_hiddens=256, ffn_num_hiddens=512, 
                        num_heads=4, num_layers=2, dropout=0.1, max_len=512, devices=devices)

    # 通过实例化SNLIBERTDataset类来生成训练和测试样本。
    # 这些样本将在自然语言推断的训练和测试期间进行小批量读取。
    batch_size, max_len, num_workers = 512, 128, 0
    data_dir = './data/snli_1.0'
    # 读取snli_1.0_train.txt并预处理(切词等)
    
    # train_set = SNLIBERTDataset(d2l.read_snli(data_dir, True), max_len, vocab)
    train_set = SNLIBERTDataset(d2l.read_snli(data_dir, False), max_len, vocab) # 应为True 
    # 读取snli_1.0_test.txt并预处理(切词等)
    test_set = SNLIBERTDataset(d2l.read_snli(data_dir, False), max_len, vocab)

    train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size, num_workers=num_workers)
    # 预训练的BERT模型bert被送到用于下游应用的BERTClassifier实例net中
    net = BERTClassifier(bert)
    net.load_state_dict(torch.load("./fine-tuning-parameter.pt")) # 只加载参数
    lr, num_epochs = 1e-4, 1
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss(reduction='none')
    # 已经使用预训练模型,不用再训练?
    print(len(train_iter))
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
    
    torch.save(net.state_dict(), "./fine-tuning-parameter.pt") # 只保存参数
