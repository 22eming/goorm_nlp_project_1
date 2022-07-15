import os
import os.path
import pandas as pd

from transformers import BertTokenizer
from utils import make_id_file, make_id_file_test, SentimentDataset, SentimentTestDataset, collate_fn_style, collate_fn_style_test

def data2dataset():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_pos, train_neg, dev_pos, dev_neg = make_id_file('yelp', tokenizer)
    train_dataset = SentimentDataset(tokenizer, train_pos, train_neg)
    dev_dataset = SentimentDataset(tokenizer, dev_pos, dev_neg)
    
    test_df = pd.read_csv('test_no_label.csv')
    test_dataset = test_df['Id']
    test = make_id_file_test(tokenizer, test_dataset)
    test_dataset = SentimentTestDataset(tokenizer, test)

    return train_dataset, dev_dataset, test_dataset, test_df
