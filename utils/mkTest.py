# 用以将测试集格式化label-studio风格，可以用来审核测试集
import json
import random

# utils/datasets/datasets/test.json  'datasets/my_data/test_data.json
with open('datasets/my_data/test_data.json', 'r') as file:
    datas = json.load(file)
r = []
for idx,data in enumerate(datas):
    text = data['text']  #正文文本
    result = []
    entity_list = data['entity_list']
    entity_ids = {}
    for entity in entity_list:
        # 依据位置制定ID表
        entity_ids[str(entity['char_span'])] = entity['id']
        result.append({'value':{
            'start': entity['char_span'][0], 
            'end': entity['char_span'][1], 
            'text': entity['text'],
            'labels':entity['type'],     
        },
        'id': entity['id'], 
        'from_name': 'label', 
        'to_name': 'text', 
        'type': 'labels', 
        'origin': 'prediction'
         })
        
    relation_list = data['relation_list']
    for relation in relation_list:
        result.append({
            'from_id': entity_ids[str(relation['subj_char_span'])],
            'to_id': entity_ids[str(relation['obj_char_span'])],
            'type': 'relation', 
            'direction': 'right', 
            'labels': [relation['predicate']]
            })
    r.append({
        "id": random.randint(10000,99999),
        "annotations": [
          {
            "id": random.randint(1000,9999),
            "result": result,
            "was_cancelled": False,
            "ground_truth": False,
            "created_at": "2023-05-05T09:11:47.308616Z",
            "updated_at": "2023-05-05T09:11:47.308616Z",
            "lead_time": 44622.122,
            "prediction": {},
            "result_count": 0,
            "task": "任务ID",
            "parent_prediction": "",
            "parent_annotation": ""
          }
        ],
        "file_upload": "",
        "drafts": [],
        "predictions": [],
        "data": {
          "text": text
        },
        "meta": {},
        "created_at": "2023-05-05T09:11:47.308616Z",
        "updated_at": "2023-05-05T09:11:47.308616Z",
        "project": idx # 自增ID
      })
with open("utils/datasets/examine/examine_test.json", "w") as file:
    json.dump(r, file,indent=4)


