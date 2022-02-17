# -*- coding: utf-8 -*-
# @Time : 2021/1/11 9:09
# @Author : yx
# @File : bert_sst2.py

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
from transformers import BertModel
from tqdm import tqdm
import os
import time
from transformers import BertTokenizer
from transformers import logging

logging.set_verbosity_error()




# 通过继承nn.Module类自定义符合自己需求的模型
class BertSST2Model(nn.Module):

    # 初始化类
    def __init__(self, class_size, pretrained_name='bert-base-chinese'):
        """
        Args: 
            class_size  :指定分类模型的最终类别数目，以确定线性分类器的映射维度
            pretrained_name :用以指定bert的预训练模型
        """
        super(BertSST2Model, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_name,
                                              return_dict=True)
        self.classifier = nn.Linear(768, class_size)

    def forward(self, input_ids, input_tyi, input_attn_mask):
        # print(input_ids.device)
        output = self.bert(input_ids, input_tyi, input_attn_mask)
        categories_numberic = self.classifier(output.pooler_output)
        return categories_numberic


def save_pretrained(model, path):
    # 保存模型，先利用os模块创建文件夹，后利用torch.save()写入模型文件
    os.makedirs(path, exist_ok=True)
    torch.save(model, os.path.join(path, 'model.pth'))


def load_sentence_polarity(data_path, train_ratio=0.8):
    all_data = []
    # categories用于统计分类标签的总数，用set结构去重
    categories = set()
    with open(data_path, 'r', encoding="utf8") as file:
        for sample in file.readlines():
            polar, sent = sample.strip().split("\t")
            categories.add(polar)
            all_data.append((polar, sent))
    length = len(all_data)
    train_len = int(length * train_ratio)
    train_data = all_data[:train_len]
    test_data = all_data[train_len:]
    return train_data, test_data, categories

class BertDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.data_size = len(dataset)

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
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
train_dataloader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              collate_fn=coffate_fn,
                              shuffle=True)
test_dataloader = DataLoader(test_dataset,
                             batch_size=1,
                             collate_fn=coffate_fn)


pretrained_model_name = 'bert-base-uncased'
# 创建模型 BertSST2Model
model = BertSST2Model(len(categories), pretrained_model_name)
tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
optimizer = Adam(model.parameters(), learning_rate)  #使用Adam优化器
CE_loss = nn.CrossEntropyLoss()  # 使用crossentropy作为二分类任务的损失函数

os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
device_ids = range(torch.cuda.device_count())
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 记录当前训练时间，用以记录日志和存储
timestamp = time.strftime("%m_%d_%H_%M", time.localtime())

model = torch.nn.DataParallel(model.cuda(), device_ids=device_ids,output_device=device_ids[0])

for epoch in range(1, num_epoch + 1):
    # 记录当前epoch的总loss
    total_loss = 0
    # tqdm用以观察训练进度，在console中会打印出进度条
    model.train()
    for inputs, targets in tqdm(train_dataloader, desc=f"Training Epoch {epoch}"):

        input_ids, input_tyi, input_attn_mask = inputs['input_ids'], inputs['token_type_ids'], inputs['attention_mask']
        input_ids=input_ids.cuda()
        input_attn_mask=input_attn_mask.cuda()
        input_tyi=input_tyi.cuda()
        targets=targets.cuda()

        # 清除现有的梯度
        optimizer.zero_grad()

        # 模型前向传播，model(inputs)等同于model.forward(inputs)
        bert_output = model(input_ids,input_tyi,input_attn_mask)

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
    for inputs, targets in tqdm(test_dataloader, desc=f"Testing"):
        with torch.no_grad():
            input_ids, input_tyi, input_attn_mask = inputs['input_ids'], inputs['token_type_ids'], inputs['attention_mask']
            input_ids=input_ids.cuda()
            input_attn_mask=input_attn_mask.cuda()
            input_tyi=input_tyi.cuda()
            targets=targets.cuda()

            bert_output = model(input_ids,input_tyi,input_attn_mask)
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
        checkpoints_dirname = "bert_sst2_dp_" + timestamp
        os.makedirs(checkpoints_dirname, exist_ok=True)
        save_pretrained(model.module,
                        checkpoints_dirname + '/checkpoints-{}/'.format(epoch))
