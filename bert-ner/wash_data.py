from datasets import load_dataset, load_metric
from tqdm import tqdm
import json

# 加载CoNLL-2003数据集、分词器
dataset = load_dataset('conll2003')
dataset_dict={}
label_list = dataset["train"].features["ner_tags"].feature.names

# 将训练集转换为可训练的特征形式
def tokenize_and_align_labels(examples):
    for id,tokens,ner_tags in zip(examples['id'],examples['tokens'],examples['ner_tags']):
        ner_labels=[label_list[ett_order] for ett_order in ner_tags]
        dataset_dict[id]={'sents':' '.join(tokens),'ner_labels':' '.join(ner_labels)} # ,'ner_ids':str(ner_tags)

# 获取标签列表，并加载预训练模型

datasets = dataset.map(tokenize_and_align_labels, batched=True, load_from_cache_file=False)
with open('conll2003.jsonl','w',encoding='utf8') as file:
    file.write(json.dumps(dataset_dict,indent=4))