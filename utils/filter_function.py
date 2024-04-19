# 2024年3月28日 
# 目的：测试时不纳入function，解决方案，将测试集中与function相关的实体和关系删除掉

import json



with open('datasets/my_data/test_data.json', 'r') as file:
    datas = json.load(file)

results = []

for data in datas:
    entity_list = data['entity_list']
    relation_list = data['relation_list']
    new_entity_list = list()
    new_relation_list = list()
    temp_text = []
    for entity in entity_list:
        if entity['type'][0] != "Function":
            new_entity_list.append(entity)
        else:
            temp_text.append(entity['text'])

    for relation in relation_list:
        if relation['subject'] not in temp_text and relation['object'] not in temp_text:
            new_relation_list.append(relation)
    if len(new_relation_list) == 0:
        continue
    results.append({
        "text" :data['text'],
        "id" :data['id'],
        "entity_list":new_entity_list,
        "relation_list":new_relation_list
         }
    )
with open("utils/datasets/test_325/test_filter_function.json", "w") as file:
    json.dump(results, file,indent=4)