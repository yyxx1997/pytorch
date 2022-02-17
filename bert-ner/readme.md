# 基于BERT实现简单的NER任务
项目链接：
```
https://github.com/yyxx1997/pytorch/tree/master/bert-ner
```
# 任务简介

命名实体识别（Named Entity Recognition，简称NER），又称作“专名识别”，是指识别文本中具有特定意义的实体，主要包括人名、地名、机构名、专有名词等。作用命名实体识别是信息提取、问答系统、句法分析、机器翻译、面向Semantic Web的元数据标注等应用领域的重要基础工具，在自然语言处理技术走向实用化的过程中占有重要地位。
# 数据准备
数据集采用``conll2003``的子集，总计14040条。conll2003 是命名实体中最常见的公开数据集，[官网](https://www.clips.uantwerpen.be/conll2003/ner/)有很详细的介绍。
## 获取数据集
网上关于conll2003的资源很多，去官网也可以直接下载。在这里，我图方便通过Python的第三方库``datasets``直接获取conll2003的全部数据（目前需要科学上网）并把他们抽取并保存成我需要的形式：
~~~python
from datasets import load_dataset
import json

# 加载CoNLL-2003数据集、分词器
dataset = load_dataset('conll2003')
dataset_dict={}
label_list = dataset["train"].features["ner_tags"].feature.names

# 将训练集转换为可训练的特征形式
def tokenize_and_align_labels(examples):
    for id,tokens,ner_tags in zip(examples['id'],examples['tokens'],examples['ner_tags']):
        ner_labels=[label_list[ett_order] for ett_order in ner_tags]
        dataset_dict[id]={'sents':' '.join(tokens),'ner_labels':' '.join(ner_labels)}

datasets = dataset.map(tokenize_and_align_labels, batched=True, load_from_cache_file=False)
with open('conll2003.jsonl','w',encoding='utf8') as file:
    file.write(json.dumps(dataset_dict,indent=4))
~~~
这里直接给出抽取结果：[conll2003.jsonl](https://github.com/yyxx1997/pytorch/blob/master/bert-ner/conll2003.jsonl)

## 示例
~~~json
{
	"0": {
        "sents": "SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT .",
        "ner_labels": "O O B-LOC O O O O B-PER O O O O"
    },
    "1": {
        "sents": "Nadim Ladki",
        "ner_labels": "B-PER I-PER"
    },
    "2": {
        "sents": "AL-AIN , United Arab Emirates 1996-12-06",
        "ner_labels": "B-LOC O B-LOC I-LOC I-LOC O"
    },
    "3": {
        "sents": "Japan began the defence of their Asian Cup title with a lucky 2-1 win against Syria in a Group C championship match on Friday .",
        "ner_labels": "B-LOC O O O O O B-MISC I-MISC O O O O O O O B-LOC O O O O O O O O O"
    }
 }
~~~
数据集中共有以下几张标签，在模型最终的分类结果里，我们把它们映射为对应的数字标签：
~~~json
    {
    'O': 0,
    'B-PER': 1,
    'I-PER': 2,
    'B-ORG': 3,
    'I-ORG': 4,
    'B-LOC': 5,
    'I-LOC': 6,
    'B-MISC': 7,
    'I-MISC': 8
    }
~~~
关于每个标签所代表的具体含义，请自行查阅。
## 数据加载
在这里并不体现参数调优的过程，只设置训练集和测试集，没有验证集。
```python
def load_sentence_nertags(data_path, train_ratio=0.8):
    # 本任务中暂时只用train、test做划分，不包含dev验证集，
    # train的比例由train_ratio参数指定，train_ratio=0.8代表训练语料占80%，test占20%
    # 本函数只适用于读取指定文件，不具通用性，仅作示范
    all_data = []
    with open(data_path, 'r', encoding="utf8") as file:
        res_dict = json.load(file)
    for id, item in res_dict.items():
        sent = item['sents'].split(' ')
        ner_labels = item['ner_labels'].split(' ')
        assert len(sent) == len(
            ner_labels
        ), "for every sentence,its ner_tag need to have the same length!"
        all_data.append((sent, ner_labels))
    length = len(all_data)
    train_len = int(length * train_ratio)
    train_data = all_data[:train_len]
    test_data = all_data[train_len:]
    return train_data, test_data
```
**定义Dataset和Dataloader为后续模型提供数据：**
```python
class BertDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.data_size = len(dataset)

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        # 这里可以自行定义，Dataloader会使用__getitem__(self, index)获取数据
        # 这里我设置 self.dataset[index] 规定了数据是按序号取得，序号是多少DataLoader自己算，用户不用操心
        return self.dataset[index]

data_path = "./conll2003.jsonl"  # 数据所在地址
# 获取训练、测试数据
train_data, test_data = load_sentence_nertags(data_path=data_path,
                                              train_ratio=train_ratio)
# 将训练数据和测试数据的列表封装成Dataset以供DataLoader加载
train_dataset = BertDataset(train_data)
test_dataset = BertDataset(test_data)
train_dataloader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              collate_fn=coffate_fn,
                              shuffle=True)
test_dataloader = DataLoader(test_dataset,
                             batch_size=1,
                             collate_fn=coffate_fn,
                             shuffle=True)
```
**DataLoader主要有以下几个参数：**
Args:

 - dataset (Dataset): dataset from which to load the data.
 - batch_size (int, optional): how many samples per batch to load(default: ``1``).
 - shuffle (bool, optional): set to ``True`` to have the data reshuffled at every epoch (default: ``False``).
 - collate_fn : 传入一个处理数据的回调函数

**DataLoader工作流程：**
1. 先从dataset中取出batch_size个数据
2. 对每个batch，执行``collate_fn``传入的函数以改变成为适合模型的输入
3. 下个epoch取数据前先对当前的数据集进行shuffle，以防模型学会数据的顺序而导致过拟合

有关Dataset和Dataloader具体可参考文章：[Pytorch入门：DataLoader 和 Dataset](https://blog.csdn.net/zw__chen/article/details/82806900)

\
**有关回调函数``collate_fn``:**
~~~python
# 加载预训练模型对应的tokenizer
pretrained_model_name = 'bert-base-uncased'
tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name)

def coffate_fn(examples):
    sents, all_labels = [], []
    for sent, ner_labels in examples:
        sents.append(sent)
        all_labels.append([categories[label] for label in ner_labels])
    tokenized_inputs = tokenizer(sents,
                                 truncation=True,
                                 padding=True,
                                 return_offsets_mapping=True,
                                 is_split_into_words=True,
                                 max_length=512,
                                 return_tensors="pt")
    targets = []
    for i, labels in enumerate(all_labels):
        label_ids = []
        for word_idx in tokenized_inputs.word_ids(batch_index=i):
            # 将特殊符号的标签设置为-100，以便在计算损失函数时自动忽略
            if word_idx is None:
                label_ids.append(-100)
            else:
                # 把标签设置到每个词的第一个token上
                label_ids.append(labels[word_idx])
        targets.append(label_ids)
    targets = torch.tensor(targets)
    return tokenized_inputs, targets
~~~
通过设置``return_offsets_mapping=True``，``tokenized_inputs``会具有``word_ids``属性，通过这个值我们可以得到某个切分后的词是否跟前一个词构成整体。由于BERT的词表会将一些词切分为后缀形式，所以在tokenize过程中我们需要记住这些词被切分成了哪几个词，然后把他们都进行统一标记。
假如，Hacker代表B-PER，Hacker在tokenize之后可能分解为：['Hack','##er']，那么我们需要做的就是将输出结果对应地标记为['B-PER','B-PER']。
# 模型介绍
本文采用最简单的**BertModel**，预训练模型加载的是 **bert-base-uncased**，在此基础上外加**Linear**层对句子中每个单词所对应的最后一层隐层进行映射：
```python
# 通过继承nn.Module类自定义符合自己需求的模型
class BertNERModel(nn.Module):

    # 初始化类
    def __init__(self, ner_labels, pretrained_name='bert-base-uncased'):
        """
        Args: 
            class_size  :指定分类模型的最终类别数目，以确定线性分类器的映射维度
            pretrained_name :用以指定bert的预训练模型
        """
        super(BertNERModel, self).__init__()
        # 加载HuggingFace的BertModel
        # BertModel的最终输出维度默认为768
        # return_dict=True 可以使BertModel的输出具有dict属性，即以 bert_output['last_hidden_state'] 方式调用
        self.bert = BertModel.from_pretrained(pretrained_name,
                                              return_dict=True)
        # 通过一个线性层将标签对应的维度：768->class_size
        self.classifier = nn.Linear(768, ner_labels)

    def forward(self, inputs):
        # 获取DataLoader中已经处理好的输入数据：
        # input_ids :tensor类型，shape=batch_size*max_len   max_len为当前batch中的最大句长
        # input_tyi :tensor类型，
        # input_attn_mask :tensor类型，因为input_ids中存在大量[Pad]填充，attention mask将pad部分值置为0，让模型只关注非pad部分
        input_ids, input_tyi, input_attn_mask = inputs['input_ids'], inputs[
            'token_type_ids'], inputs['attention_mask']
        output = self.bert(input_ids, input_tyi, input_attn_mask)
        # bert_output 分为两个部分：
        #   last_hidden_state:最后一个隐层的值
        #   pooler output:对应的是[CLS]的输出,用于分类任务
        # categories_numberic：tensor类型，shape=batch_size*class_size，用于后续的CrossEntropy计算
        categories_numberic = self.classifier(output.last_hidden_state)
        batch_size, seq_len, ner_class_num = categories_numberic.shape
        categories_numberic = categories_numberic.view(
            (batch_size * seq_len, ner_class_num))
        return categories_numberic
```

模型整体效果图如下（图片来源：网络）：
![BERT-NER](https://img-blog.csdnimg.cn/5583d8d8c90845cc80b84fee0f318b5e.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5aKo6I-y5piv5LiA5Y-q5Za1,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

由图中可以看出，输入在经过12个层之后，利用每个token对应地隐层输出完成最终的分类任务。

# Finetune过程
## 参数设定
训练准备阶段，设置超参数和全局变量
```
batch_size = 32
num_epoch = 5  # 训练轮次
check_step = 1  # 用以训练中途对模型进行检验：每check_step个epoch进行一次测试和保存模型
data_path = "./conll2003.jsonl"  # 数据所在地址
train_ratio = 0.8  # 训练集比例
learning_rate = 1e-5  # 优化器的学习率
```
## 优化器和损失函数
``ignore_index``用于标记在损失计算时应该跳过的位置，在本任务中，特殊标签如【CLS】、【SEP】、【PAD】等被赋值为-100，都不计入损失。
~~~
optimizer = Adam(model.parameters(), learning_rate)  #使用Adam优化器
CE_loss = nn.CrossEntropyLoss(ignore_index=-100)  # 使用crossentropy作为分类任务的损失函数
~~~
## 训练
```python
model.train()
for epoch in range(1, num_epoch + 1):
    # 记录当前epoch的总loss
    total_loss = 0
    # tqdm用以观察训练进度，在console中会打印出进度条

    for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch}"):
        # tqdm(train_dataloader, desc=f"Training Epoch {epoch}") 会自动执行DataLoader的工作流程，
        # 想要知道内部如何工作可以在debug时将断点打在 coffate_fn 函数内部，查看数据的处理过程

        # 对batch中的每条tensor类型数据，都执行.to(device)，
        # 因为模型和数据要在同一个设备上才能运行
        inputs, targets = [x.to(device) for x in batch]
        targets = targets.view(-1)
        # 清除现有的梯度
        optimizer.zero_grad()

        # 模型前向传播，model(inputs)等同于model.forward(inputs)
        bert_output = model(inputs)

        # 计算损失，交叉熵损失计算可参考：https://zhuanlan.zhihu.com/p/159477597
        loss = CE_loss(bert_output, targets)

        # 梯度反向传播
        loss.backward()

        # 根据反向传播的值更新模型的参数
        optimizer.step()

        # 统计总的损失，.item()方法用于取出tensor中的值
        total_loss += loss.item()
```
## 测试
通过argmax函数将模型预测的序列转化为对应的标签，用于评估函数的计算：
~~~python
	target_labels = []
    pred_labels = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc=f"Testing"):
            inputs, targets = [x.to(device) for x in batch]
            targets = targets.view(-1)
            bert_output = model(inputs)
            predictions = bert_output.argmax(dim=-1)
            target_labels += [categories[i]
                              for i in targets.tolist() if i != -100]
            pred_labels += [
                categories[i] for i in predictions.tolist()[1:-1] if i != -100
            ]

    precision, recall, f1 = evaluate(real_label=target_labels,
                                     predict_label=pred_labels)
    print("precision is {}\nrecall is {}\nf1 is {}".format(
        precision, recall, f1))
~~~
以下函数用于序列标注的准确率和召回率计算，详情查看：https://zhuanlan.zhihu.com/p/56582082
```python
def split_entity(label_sequence):
    entity_mark = dict()
    entity_pointer = None
    for index, label in enumerate(label_sequence):
        if label.startswith('B'):
            category = label.split('-')[1]
            entity_pointer = (index, category)
            entity_mark.setdefault(entity_pointer, [label])
        elif label.startswith('I'):
            if entity_pointer is None: continue
            if entity_pointer[1] != label.split('-')[1]: continue
            entity_mark[entity_pointer].append(label)
        else:
            entity_pointer = None
    return entity_mark


def evaluate(real_label, predict_label):
    # 序列标注的准确率和召回率计算，详情查看：https://zhuanlan.zhihu.com/p/56582082
    real_entity_mark = split_entity(real_label)
    predict_entity_mark = split_entity(predict_label)

    true_entity_mark = dict()
    key_set = real_entity_mark.keys() & predict_entity_mark.keys()
    for key in key_set:
        real_entity = real_entity_mark.get(key)
        predict_entity = predict_entity_mark.get(key)
        if tuple(real_entity) == tuple(predict_entity):
            true_entity_mark.setdefault(key, real_entity)

    real_entity_num = len(real_entity_mark)
    predict_entity_num = len(predict_entity_mark)
    true_entity_num = len(true_entity_mark)

    precision = true_entity_num / predict_entity_num
    recall = true_entity_num / real_entity_num
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1
```
## 运行结果
模型在数据集上的F1值由0.2上升到0.7左右，有巨大提高。
# 完整代码
```python
# -*- coding: utf-8 -*-
# @Time : 2021/1/11 9:09
# @Author : yx
# @File : bert_ner.py

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel
from tqdm import tqdm
import os
import time
import json
from transformers import BertTokenizerFast
from transformers import logging

# 设置transformers模块的日志等级，减少不必要的警告，对训练过程无影响，请忽略
logging.set_verbosity_error()

# 环境变量：设置程序能使用的GPU序号。例如：
# 当前服务器有8张GPU可用，想用其中的第2、5、8卡，这里应该设置为:
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,4,7"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 通过继承nn.Module类自定义符合自己需求的模型
class BertNERModel(nn.Module):

    # 初始化类
    def __init__(self, ner_labels, pretrained_name='bert-base-uncased'):
        """
        Args: 
            class_size  :指定分类模型的最终类别数目，以确定线性分类器的映射维度
            pretrained_name :用以指定bert的预训练模型
        """
        super(BertNERModel, self).__init__()
        # 加载HuggingFace的BertModel
        # BertModel的最终输出维度默认为768
        # return_dict=True 可以使BertModel的输出具有dict属性，即以 bert_output['last_hidden_state'] 方式调用
        self.bert = BertModel.from_pretrained(pretrained_name,
                                              return_dict=True)
        # 通过一个线性层将标签对应的维度：768->class_size
        self.classifier = nn.Linear(768, ner_labels)

    def forward(self, inputs):
        # 获取DataLoader中已经处理好的输入数据：
        # input_ids :tensor类型，shape=batch_size*max_len   max_len为当前batch中的最大句长
        # input_tyi :tensor类型，
        # input_attn_mask :tensor类型，因为input_ids中存在大量[Pad]填充，attention mask将pad部分值置为0，让模型只关注非pad部分
        input_ids, input_tyi, input_attn_mask = inputs['input_ids'], inputs[
            'token_type_ids'], inputs['attention_mask']
        output = self.bert(input_ids, input_tyi, input_attn_mask)
        # bert_output 分为两个部分：
        #   last_hidden_state:最后一个隐层的值
        #   pooler output:对应的是[CLS]的输出,用于分类任务
        # categories_numberic：tensor类型，shape=batch_size*class_size，用于后续的CrossEntropy计算
        categories_numberic = self.classifier(output.last_hidden_state)
        batch_size, seq_len, ner_class_num = categories_numberic.shape
        categories_numberic = categories_numberic.view(
            (batch_size * seq_len, ner_class_num))
        return categories_numberic


def save_pretrained(model, path):
    # 保存模型，先利用os模块创建文件夹，后利用torch.save()写入模型文件
    os.makedirs(path, exist_ok=True)
    torch.save(model, os.path.join(path, 'model.pth'))


def load_sentence_nertags(data_path, train_ratio=0.8):
    # 本任务中暂时只用train、test做划分，不包含dev验证集，
    # train的比例由train_ratio参数指定，train_ratio=0.8代表训练语料占80%，test占20%
    # 本函数只适用于读取指定文件，不具通用性，仅作示范
    all_data = []
    with open(data_path, 'r', encoding="utf8") as file:
        res_dict = json.load(file)
    for id, item in res_dict.items():
        sent = item['sents'].split(' ')
        ner_labels = item['ner_labels'].split(' ')
        assert len(sent) == len(
            ner_labels
        ), "for every sentence,its ner_tag need to have the same length!"
        all_data.append((sent, ner_labels))
    length = len(all_data)
    train_len = int(length * train_ratio)
    train_data = all_data[:train_len]
    test_data = all_data[train_len:]
    return train_data, test_data


"""
torch提供了优秀的数据加载类Dataloader，可以自动加载数据。
1. 想要使用torch的DataLoader作为训练数据的自动加载模块，就必须使用torch提供的Dataset类
2. 一定要具有__len__和__getitem__的方法，不然DataLoader不知道如何如何加载数据
这里是固定写法，是官方要求，不懂可以不做深究，一般的任务这里都通用
"""


class BertDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.data_size = len(dataset)

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        # 这里可以自行定义，Dataloader会使用__getitem__(self, index)获取数据
        # 这里我设置 self.dataset[index] 规定了数据是按序号取得，序号是多少DataLoader自己算，用户不用操心
        return self.dataset[index]


def coffate_fn(examples):
    sents, all_labels = [], []
    for sent, ner_labels in examples:
        sents.append(sent)
        all_labels.append([categories[label] for label in ner_labels])
    tokenized_inputs = tokenizer(sents,
                                 truncation=True,
                                 padding=True,
                                 return_offsets_mapping=True,
                                 is_split_into_words=True,
                                 max_length=512,
                                 return_tensors="pt")
    targets = []
    for i, labels in enumerate(all_labels):
        label_ids = []
        for word_idx in tokenized_inputs.word_ids(batch_index=i):
            # 将特殊符号的标签设置为-100，以便在计算损失函数时自动忽略
            if word_idx is None:
                label_ids.append(-100)
            else:
                # 把标签设置到每个词的第一个token上
                label_ids.append(labels[word_idx])
        targets.append(label_ids)
    targets = torch.tensor(targets)
    return tokenized_inputs, targets


def split_entity(label_sequence):
    entity_mark = dict()
    entity_pointer = None
    for index, label in enumerate(label_sequence):
        if label.startswith('B'):
            category = label.split('-')[1]
            entity_pointer = (index, category)
            entity_mark.setdefault(entity_pointer, [label])
        elif label.startswith('I'):
            if entity_pointer is None:
                continue
            if entity_pointer[1] != label.split('-')[1]:
                continue
            entity_mark[entity_pointer].append(label)
        else:
            entity_pointer = None
    return entity_mark


def evaluate(real_label, predict_label):
    # 序列标注的准确率和召回率计算，详情查看：https://zhuanlan.zhihu.com/p/56582082
    real_entity_mark = split_entity(real_label)
    predict_entity_mark = split_entity(predict_label)

    true_entity_mark = dict()
    key_set = real_entity_mark.keys() & predict_entity_mark.keys()
    for key in key_set:
        real_entity = real_entity_mark.get(key)
        predict_entity = predict_entity_mark.get(key)
        if tuple(real_entity) == tuple(predict_entity):
            true_entity_mark.setdefault(key, real_entity)

    real_entity_num = len(real_entity_mark)
    predict_entity_num = len(predict_entity_mark)
    true_entity_num = len(true_entity_mark)

    precision = true_entity_num / predict_entity_num
    recall = true_entity_num / real_entity_num
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


# 训练准备阶段，设置超参数和全局变量

batch_size = 32
num_epoch = 5  # 训练轮次
check_step = 1  # 用以训练中途对模型进行检验：每check_step个epoch进行一次测试和保存模型
data_path = "./conll2003.jsonl"  # 数据所在地址
train_ratio = 0.8  # 训练集比例
learning_rate = 1e-5  # 优化器的学习率

# 获取训练、测试数据、分类类别总数
train_data, test_data = load_sentence_nertags(data_path=data_path,
                                              train_ratio=train_ratio)

categories = {
    'O': 0,
    'B-PER': 1,
    'I-PER': 2,
    'B-ORG': 3,
    'I-ORG': 4,
    'B-LOC': 5,
    'I-LOC': 6,
    'B-MISC': 7,
    'I-MISC': 8,
    0: 'O',
    1: 'B-PER',
    2: 'I-PER',
    3: 'B-ORG',
    4: 'I-ORG',
    5: 'B-LOC',
    6: 'I-LOC',
    7: 'B-MISC',
    8: 'I-MISC'
}

# 将训练数据和测试数据的列表封装成Dataset以供DataLoader加载
train_dataset = BertDataset(train_data)
test_dataset = BertDataset(test_data)
"""
DataLoader主要有以下几个参数：
Args:
    dataset (Dataset): dataset from which to load the data.
    batch_size (int, optional): how many samples per batch to load(default: ``1``).
    shuffle (bool, optional): set to ``True`` to have the data reshuffled at every epoch (default: ``False``).
    collate_fn : 传入一个处理数据的回调函数
DataLoader工作流程：
1. 先从dataset中取出batch_size个数据
2. 对每个batch，执行collate_fn传入的函数以改变成为适合模型的输入
3. 下个epoch取数据前先对当前的数据集进行shuffle，以防模型学会数据的顺序而导致过拟合
"""
train_dataloader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              collate_fn=coffate_fn,
                              shuffle=True)
test_dataloader = DataLoader(test_dataset,
                             batch_size=1,
                             collate_fn=coffate_fn,
                             shuffle=True)

#固定写法，可以牢记，cuda代表Gpu
# torch.cuda.is_available()可以查看当前Gpu是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练模型，因为这里是英文数据集，需要用在英文上的预训练模型：bert-base-uncased
# uncased指该预训练模型对应的词表不区分字母的大小写
# 详情可了解：https://huggingface.co/bert-base-uncased
pretrained_model_name = 'bert-base-uncased'

model = BertNERModel(len(categories), pretrained_model_name)
# 固定写法，将模型加载到device上，
# 如果是GPU上运行，此时可以观察到GPU的显存增加
model.to(device)
# 加载预训练模型对应的tokenizer
tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name)

# 训练过程
# Adam是最近较为常用的优化器，详情可查看：https://www.jianshu.com/p/aebcaf8af76e
optimizer = Adam(model.parameters(), learning_rate)  # 使用Adam优化器
CE_loss = nn.CrossEntropyLoss(ignore_index=-100)  # 使用crossentropy作为分类任务的损失函数

# 记录当前训练时间，用以记录日志和存储
timestamp = time.strftime("%m_%d_%H_%M", time.localtime())

# 开始训练，model.train()固定写法，详情可以百度
model.train()
for epoch in range(1, num_epoch + 1):
    # 记录当前epoch的总loss
    total_loss = 0
    # tqdm用以观察训练进度，在console中会打印出进度条

    for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch}"):
        # tqdm(train_dataloader, desc=f"Training Epoch {epoch}") 会自动执行DataLoader的工作流程，
        # 想要知道内部如何工作可以在debug时将断点打在 coffate_fn 函数内部，查看数据的处理过程

        # 对batch中的每条tensor类型数据，都执行.to(device)，
        # 因为模型和数据要在同一个设备上才能运行
        inputs, targets = [x.to(device) for x in batch]
        targets = targets.view(-1)
        # 清除现有的梯度
        optimizer.zero_grad()

        # 模型前向传播，model(inputs)等同于model.forward(inputs)
        bert_output = model(inputs)

        # 计算损失，交叉熵损失计算可参考：https://zhuanlan.zhihu.com/p/159477597
        loss = CE_loss(bert_output, targets)

        # 梯度反向传播
        loss.backward()

        # 根据反向传播的值更新模型的参数
        optimizer.step()

        # 统计总的损失，.item()方法用于取出tensor中的值
        total_loss += loss.item()

    #测试过程
    target_labels = []
    pred_labels = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc=f"Testing"):
            inputs, targets = [x.to(device) for x in batch]
            targets = targets.view(-1)
            bert_output = model(inputs)
            predictions = bert_output.argmax(dim=-1)
            target_labels += [categories[i]
                              for i in targets.tolist() if i != -100]
            pred_labels += [
                categories[i] for i in predictions.tolist()[1:-1] if i != -100
            ]

    precision, recall, f1 = evaluate(real_label=target_labels,
                                     predict_label=pred_labels)
    print("precision is {}\nrecall is {}\nf1 is {}".format(
        precision, recall, f1))

    if epoch % check_step == 0:
        # 保存模型
        checkpoints_dirname = "bert_ner_" + timestamp
        os.makedirs(checkpoints_dirname, exist_ok=True)
        save_pretrained(model,
                        checkpoints_dirname + '/checkpoints-{}/'.format(epoch))

```

