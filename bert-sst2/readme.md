# 基于BERT实现简单的情感分类任务
项目链接：
```
https://github.com/yyxx1997/pytorch/tree/master/bert-sst2
```
# 任务简介

**情感分类**是指根据文本所表达的含义和情感信息将文本划分成褒扬的或贬义的两种或几种类型，是对文本作者倾向性和观点、态度的划分，因此有时也称倾向性分析（opinion analysis）。

本文通过简单的情感二分类任务作为样例，展示如何利用预训练模型**BERT**进行简单的Finetune过程。
# 数据准备
此任务以演示BERT用法为主，数据集采用SST-2的子集，即在原本数据集基础上进行抽取得到的部分，总计10000条。
## SST-2数据集
SST数据集： 斯坦福大学发布的一个情感分析数据集，主要针对电影评论来做情感分类，因此SST属于单个句子的文本分类任务（其中SST-2是二分类，SST-5是五分类，SST-5的情感极性区分的更细致）

SST数据集地址：https://nlp.stanford.edu/sentiment/index.html

有关SST数据的处理部分不再赘述，这里给出抽取结果：[sst2_shuffled.tsv](https://github.com/yyxx1997/pytorch/blob/master/bert-sst2/sst2_shuffled.tsv)

## 示例
0——positive
1——negative
sentiment polarity     | sentence
-------- | -----
1 |	this is the case of a pregnant premise being wasted by a...
0 |	is office work really as alienating as 'bartleby' so effectively...
0 |	horns and halos benefits from serendipity but also reminds...
1 |	heavy-handed exercise in time-vaulting literary pretension.
0 |	easily one of the best and most exciting movies of the year.
1 |	you . . . get a sense of good intentions derailed by a failure...
1 |	johnson has , in his first film , set himself a task he is not nearly up to.
## 数据加载
在这里并不体现参数调优的过程，只设置训练集和测试集，没有验证集。
```python
def load_sentence_polarity(data_path, train_ratio=0.8):
    # 本任务中暂时只用train、test做划分，不包含dev验证集，
    # train的比例由train_ratio参数指定，train_ratio=0.8代表训练语料占80%，test占20%
    # 本函数只适用于读取指定文件，不具通用性，仅作示范
    all_data = []
    # categories用于统计分类标签的总数，用set结构去重
    categories = set()
    with open(data_path, 'r', encoding="utf8") as file:
        for sample in file.readlines():
            # polar指情感的类别，当前只有两种：
            #   ——0：positive
            #   ——1：negative
            # sent指对应的句子
            polar, sent = sample.strip().split("\t")
            categories.add(polar)
            all_data.append((polar, sent))
    length = len(all_data)
    train_len = int(length * train_ratio)
    train_data = all_data[:train_len]
    test_data = all_data[train_len:]
    return train_data, test_data, categories
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


def coffate_fn(examples):
    inputs, targets = [], []
    for polar, sent in examples:
        inputs.append(sent)
        targets.append(int(polar))
    inputs = tokenizer(inputs,
                       padding=True,
                       truncation=True,
                       return_tensors="pt",
                       max_length=512)
    targets = torch.tensor(targets)
    return inputs, targets

data_path = "sst2_shuffled.tsv"  # 数据所在地址
# 获取训练、测试数据、分类类别总数
train_data, test_data, categories = load_sentence_polarity(
    data_path=data_path, train_ratio=train_ratio)

# 将训练数据和测试数据的列表封装成Dataset以供DataLoader加载
train_dataset = BertDataset(train_data)
test_dataset = BertDataset(test_data)

train_dataloader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              collate_fn=coffate_fn,
                              shuffle=True)
test_dataloader = DataLoader(test_dataset,
                             batch_size=1,
                             collate_fn=coffate_fn)
```
DataLoader主要有以下几个参数：
Args:

 - dataset (Dataset): dataset from which to load the data.
 - batch_size (int, optional): how many samples per batch to load(default: ``1``).
 - shuffle (bool, optional): set to ``True`` to have the data reshuffled at every epoch (default: ``False``).
 - collate_fn : 传入一个处理数据的回调函数

DataLoader工作流程：
1. 先从dataset中取出batch_size个数据
2. 对每个batch，执行collate_fn传入的函数以改变成为适合模型的输入
3. 下个epoch取数据前先对当前的数据集进行shuffle，以防模型学会数据的顺序而导致过拟合

有关Dataset和Dataloader具体可参考文章：[Pytorch入门：DataLoader 和 Dataset](https://blog.csdn.net/zw__chen/article/details/82806900)
# 模型介绍
本文采用最简单的**BertModel**，预训练模型加载的是 **bert-base-uncased**，在此基础上外加**Linear**层进行线性映射达到二分类目的：
```python
from transformers import BertModel

# 通过继承nn.Module类自定义符合自己需求的模型
class BertSST2Model(nn.Module):

    # 初始化类
    def __init__(self, class_size, pretrained_name='bert-base-uncased'):
        """
        Args: 
            class_size  :指定分类模型的最终类别数目，以确定线性分类器的映射维度
            pretrained_name :用以指定bert的预训练模型
        """
        super(BertSST2Model, self).__init__()
        # 加载HuggingFace的BertModel
        # BertModel的最终输出维度默认为768
        # return_dict=True 可以使BertModel的输出可以用dict形式调用，例如 bert_output['last_hidden_state'] 获取最后的隐层
        self.bert = BertModel.from_pretrained(pretrained_name,
                                              return_dict=True)
        # 通过一个线性层将[CLS]标签对应的维度：768->class_size
        # class_size 在SST-2情感2分类任务中设置为：2
        self.classifier = nn.Linear(768, class_size)
```

模型整体效果图如下（图片来源：网络）：
![在这里插入图片描述](https://img-blog.csdnimg.cn/1009aff322304d1799299e8e19864c50.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAd2VpeGluXzQ1MTAxOTU5,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
由图中可以看出，输入在经过12个层之后，利用【CLS】标记完成最终的分类任务。但这里需要注意的是：
+ BertModel对【CLS】标签所在位置最后会经过一个Pooler池化层，所以并不是直接拿最后隐层的对应值进行的线性映射。
+ Linear层以Pooler的输出作为输入，是一般BERT分类任务的通用做法

Pooler池化层具体可参考 [transformers源码](https://github.com/huggingface/transformers/blob/f65fe3663a6c62975a9c04654703252644c9a652/src/transformers/models/bert/modeling_bert.py#L1009)。

# Finetune过程
## 参数设定
训练准备阶段，设置超参数和全局变量
```
batch_size = 16	# 同时训练的数据大小
num_epoch = 10  # 训练轮次
check_step = 2  # 用以训练中途对模型进行检验：每check_step个epoch进行一次测试和保存模型
data_path = "sst2_shuffled.tsv"  # 数据所在地址
train_ratio = 0.8  # 训练集比例
learning_rate = 1e-5  # 优化器的学习率
```
## 优化器和损失函数
~~~
optimizer = Adam(model.parameters(), learning_rate)  #使用Adam优化器
CE_loss = nn.CrossEntropyLoss()  # 使用crossentropy作为二分类任务的损失函数
~~~
## 训练
```python
model.train()
for epoch in range(1, num_epoch + 1):
    # 记录当前epoch的总loss
    total_loss = 0
    for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch}"):
        
        # 对batch中的每条tensor类型数据，都执行.to(device)，
        # 因为模型和数据要在同一个设备上才能运行
        inputs, targets = [x.to(device) for x in batch]

        # 清除现有的梯度
        optimizer.zero_grad()

        # 模型前向传播
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
```python
  # acc统计模型在测试数据上分类结果中的正确个数
  acc = 0
   for batch in tqdm(test_dataloader, desc=f"Testing"):
       inputs, targets = [x.to(device) for x in batch]
       with torch.no_grad():
           bert_output = model(inputs)
           """
           .argmax()用于取出一个tensor向量中的最大值对应的下表序号，dim指定了维度
           假设 bert_output为3*2的tensor：
           tensor
           [
               [3.2,1.1],
               [0.4,0.6],
               [-0.1,0.2]
           ]
           则 bert_output.argmax(dim=1) 的结果为：tensor[0,1,1]
           """
           acc += (bert_output.argmax(dim=1) == targets).sum().item()
   #输出在测试集上的准确率
   print(f"Acc: {acc / len(test_dataloader):.2f}")
```
## 运行结果
模型在数据集上的准确率由50%以下上升到85%左右，有明显提升。
# 完整代码
```python
# -*- coding: utf-8 -*-
# @Time : 2021/1/11 9:09
# @Author : yx
# @File : bert_sst2.py

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel
from tqdm import tqdm
import os
import time
from transformers import BertTokenizer
from transformers import logging

# 设置transformers模块的日志等级，减少不必要的警告，对训练过程无影响，请忽略
logging.set_verbosity_error()

# 环境变量：设置程序能使用的GPU序号。例如：
# 当前服务器有8张GPU可用，想用其中的第2、5、8卡，这里应该设置为:
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,4,7"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 通过继承nn.Module类自定义符合自己需求的模型
class BertSST2Model(nn.Module):

    # 初始化类
    def __init__(self, class_size, pretrained_name='bert-base-chinese'):
        """
        Args: 
            class_size  :指定分类模型的最终类别数目，以确定线性分类器的映射维度
            pretrained_name :用以指定bert的预训练模型
        """
        # 类继承的初始化，固定写法
        super(BertSST2Model, self).__init__()
        # 加载HuggingFace的BertModel
        # BertModel的最终输出维度默认为768
        # return_dict=True 可以使BertModel的输出具有dict属性，即以 bert_output['last_hidden_state'] 方式调用
        self.bert = BertModel.from_pretrained(pretrained_name,
                                              return_dict=True)
        # 通过一个线性层将[CLS]标签对应的维度：768->class_size
        # class_size 在SST-2情感分类任务中设置为：2
        self.classifier = nn.Linear(768, class_size)

    def forward(self, inputs):
        # 获取DataLoader中已经处理好的输入数据：
        # input_ids :tensor类型，shape=batch_size*max_len   max_len为当前batch中的最大句长
        # input_tyi :tensor类型，
        # input_attn_mask :tensor类型，因为input_ids中存在大量[Pad]填充，attention mask将pad部分值置为0，让模型只关注非pad部分
        input_ids, input_tyi, input_attn_mask = inputs['input_ids'], inputs[
            'token_type_ids'], inputs['attention_mask']
        # 将三者输入进模型，如果想知道模型内部如何运作，前面的蛆以后再来探索吧~
        output = self.bert(input_ids, input_tyi, input_attn_mask)
        # bert_output 分为两个部分：
        #   last_hidden_state:最后一个隐层的值
        #   pooler output:对应的是[CLS]的输出,用于分类任务
        # 通过线性层将维度：768->2
        # categories_numberic：tensor类型，shape=batch_size*class_size，用于后续的CrossEntropy计算
        categories_numberic = self.classifier(output.pooler_output)
        return categories_numberic


def save_pretrained(model, path):
    # 保存模型，先利用os模块创建文件夹，后利用torch.save()写入模型文件
    os.makedirs(path, exist_ok=True)
    torch.save(model, os.path.join(path, 'model.pth'))


def load_sentence_polarity(data_path, train_ratio=0.8):
    # 本任务中暂时只用train、test做划分，不包含dev验证集，
    # train的比例由train_ratio参数指定，train_ratio=0.8代表训练语料占80%，test占20%
    # 本函数只适用于读取指定文件，不具通用性，仅作示范
    all_data = []
    # categories用于统计分类标签的总数，用set结构去重
    categories = set()
    with open(data_path, 'r', encoding="utf8") as file:
        for sample in file.readlines():
            # polar指情感的类别，当前只有两种：
            #   ——0：positive
            #   ——1：negative
            # sent指对应的句子
            polar, sent = sample.strip().split("\t")
            categories.add(polar)
            all_data.append((polar, sent))
    length = len(all_data)
    train_len = int(length * train_ratio)
    train_data = all_data[:train_len]
    test_data = all_data[train_len:]
    return train_data, test_data, categories


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
    inputs, targets = [], []
    for polar, sent in examples:
        inputs.append(sent)
        targets.append(int(polar))
    inputs = tokenizer(inputs,
                       padding=True,
                       truncation=True,
                       return_tensors="pt",
                       max_length=512)
    targets = torch.tensor(targets)
    return inputs, targets


# 训练准备阶段，设置超参数和全局变量

batch_size = 32
num_epoch = 5  # 训练轮次
check_step = 1  # 用以训练中途对模型进行检验：每check_step个epoch进行一次测试和保存模型
data_path = "./sst2_shuffled.tsv"  # 数据所在地址
train_ratio = 0.8  # 训练集比例
learning_rate = 1e-5  # 优化器的学习率

# 获取训练、测试数据、分类类别总数
train_data, test_data, categories = load_sentence_polarity(
    data_path=data_path, train_ratio=train_ratio)

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
                             collate_fn=coffate_fn)

#固定写法，可以牢记，cuda代表Gpu
# torch.cuda.is_available()可以查看当前Gpu是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练模型，因为这里是英文数据集，需要用在英文上的预训练模型：bert-base-uncased
# uncased指该预训练模型对应的词表不区分字母的大小写
# 详情可了解：https://huggingface.co/bert-base-uncased
pretrained_model_name = 'bert-base-uncased'
# 创建模型 BertSST2Model
model = BertSST2Model(len(categories), pretrained_model_name)
# 固定写法，将模型加载到device上，
# 如果是GPU上运行，此时可以观察到GPU的显存增加
model.to(device)
# 加载预训练模型对应的tokenizer
tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)

# 训练过程
# Adam是最近较为常用的优化器，详情可查看：https://www.jianshu.com/p/aebcaf8af76e
optimizer = Adam(model.parameters(), learning_rate)  #使用Adam优化器
CE_loss = nn.CrossEntropyLoss()  # 使用crossentropy作为二分类任务的损失函数

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
    # acc统计模型在测试数据上分类结果中的正确个数
    acc = 0
    for batch in tqdm(test_dataloader, desc=f"Testing"):
        inputs, targets = [x.to(device) for x in batch]
        # with torch.no_grad(): 为固定写法，
        # 这个代码块中的全部有关tensor的操作都不产生梯度。目的是节省时间和空间，不加也没事
        with torch.no_grad():
            bert_output = model(inputs)
            """
            .argmax()用于取出一个tensor向量中的最大值对应的下表序号，dim指定了维度
            假设 bert_output为3*2的tensor：
            tensor
            [
                [3.2,1.1],
                [0.4,0.6],
                [-0.1,0.2]
            ]
            则 bert_output.argmax(dim=1) 的结果为：tensor[0,1,1]
            """
            acc += (bert_output.argmax(dim=1) == targets).sum().item()
    #输出在测试集上的准确率
    print(f"Acc: {acc / len(test_dataloader):.2f}")

    if epoch % check_step == 0:
        # 保存模型
        checkpoints_dirname = "bert_sst2_" + timestamp
        os.makedirs(checkpoints_dirname, exist_ok=True)
        save_pretrained(model,
                        checkpoints_dirname + '/checkpoints-{}/'.format(epoch))

```

