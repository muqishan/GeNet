'''用于将某些类别、实体合并'''

import json

def adjust_data(data):
    """根据提供的策略调整数据"""
    valid_entities = ['Gene', 'Cancer', 'Signal pathway', 'Function', 'MultiGene', 'MultiCancer']
    discard_relations = ['gene:pathway', 'gene:related', 'gene:positively related', 'gene:negatively related', 
                         'gene:transcriptional coactivation', 'gene:dependence', 'gene:inhibits dependence']

    new_data = []
    for article in data:
        new_annotations = []
        for annotation in article.get('annotations', []):
            new_results = []
            for result in annotation.get('result', []):
                # 调整实体
                if 'value' in result and 'labels' in result['value']:
                    if result['value']['labels'][0] not in valid_entities:
                        continue
                    if result['value']['labels'][0] in ['MultiGene', 'MultiCancer']:
                        result['value']['labels'][0] = result['value']['labels'][0].replace('Multi', '')
                    new_results.append(result)
                # 调整关系
                elif 'from_id' in result and 'labels' in result and result['labels']:
                    if result['labels'][0] in discard_relations:
                        continue
                    if result['labels'][0] == 'gene:target':
                        result['labels'][0] = 'gene:upstream'
                    new_results.append(result)
            if new_results:
                
                annotation['result'] = new_results
                new_annotations.append(annotation)
        if new_annotations:
            for _ in new_annotations:
                # new_annotations[0]['completed_by'] = ''
                try:
                    new_annotations[0].pop('completed_by')
                except Exception as E:
                    pass
            article['annotations'] = new_annotations
            new_data.append(article)
    return new_data

# 读取JSON文件
with open('utils/datasets/new_datasets.json', 'r') as file:
    data = json.load(file)

# 调整数据
adjusted_data = adjust_data(data)

# 保存调整后的数据为新的JSON文件
with open('utils/datasets/new_datasets.json', 'w') as file:
    json.dump(adjusted_data, file, indent=4)

print("Data has been adjusted and saved as 'adjusted_data.json'.")
