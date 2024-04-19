'''
用以判断标注的位置是否准确
'''

import os
import json
import random
from transformers import (BertTokenizerFast)


# added_token = [f"[unused{i}]" for i in range(1, 17)]
tokenizer = BertTokenizerFast.from_pretrained(
    "pubmedbert", do_basic_tokenize=False)


data_paths = [os.path.join('utils/datasets/datasets_tokens',i)  for i in os.listdir('utils/datasets/datasets_tokens')]
data = []
for dataset_path in data_paths:
    with open('utils/datasets/datasets_tokens/test.json', "r") as file:
        data = data + (json.load(file))

print()
paper = data[random.randint(0, len(data))]
print()
text1 = 'Histone demethylase JMJD2D promotes the self-renewal of liver cancer stem-like cells by enhancing EpCAM and Sox9 expression.'
for paper in data:
    sentece_tokens = tokenizer.tokenize(paper['text'])
    if paper['entity_list']:
        for entitys in paper['entity_list']:
            # print('text（标注名）:',entitys['text'])
            # print('字符级位置对应值',paper['text'][entitys['char_span'][0]:entitys['char_span'][1]])
            # print('token级位置对应值:',sentece_tokens[entitys['tok_span'][0]:entitys['tok_span'][1]])
            # print('-----------------------------------------------------------')
            a = entitys['text'].replace(' ','')
            b = paper['text'][entitys['char_span'][0]:entitys['char_span'][1]].replace(' ','')
            c = ''.join(sentece_tokens[entitys['tok_span'][0]:entitys['tok_span'][1]]).replace(' ','').replace('#','')
            if not (a.lower() == b.lower() and a.lower() == c.lower() and b.lower() == c.lower()):
                print(a.lower())
                print(b.lower())
                print(c.lower())
                print(f"---------{entitys['id']}--------")
            


    # print('-----------------------------------------------------------')
    if paper['relation_list']:
        for relation_list in paper['relation_list']:
            pass
            # print('主体字符级位置  :',paper['text'][relation_list['subj_char_span'][0]:relation_list['subj_char_span'][1]])
            # print('主体token级位置 :',sentece_tokens[relation_list['subj_tok_span'][0]:relation_list['subj_tok_span'][1]])
            # print('客体字符级位置  :',paper['text'][relation_list['obj_char_span'][0]:relation_list['obj_char_span'][1]])
            # print('客体token级位置 :',sentece_tokens[relation_list['obj_tok_span'][0]:relation_list['obj_tok_span'][1]])
            # print('-----------------------------------------------------------')