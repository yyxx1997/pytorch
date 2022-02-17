
def load_sentence_polarity(data_path,train_ratio=0.8):
    all_data=[]
    categories=set()
    with open(data_path,'r',encoding="utf8") as file:
        for sample in file.readlines():
            polar,sent=sample.strip().split("\t")
            categories.add(polar)
            all_data.append((polar,sent))
    length=len(all_data)
    train_len=int(length*train_ratio)
    train_data=all_data[:train_len]
    test_data=all_data[train_len:]
    return train_data,test_data,categories

if __name__=="__main__":
    data_path="sst2_shuffled.tsv"
    load_sentence_polarity(data_path)