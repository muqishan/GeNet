import json
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import codecs

from transformers import (BertTokenizerFast)

# 用以将label-studio数据风格，转化为mydata风格

# added_token = [f"[unused{i}]" for i in range(1, 17)]
tokenizer = BertTokenizerFast.from_pretrained(
    "pubmedbert", do_basic_tokenize=False)

def get_entities_in_same_sentence(text, sentence_list, annotations):
    """Determine which entities are in the same sentence."""
    
    # Calculate the start and end character positions for each sentence
    sentence_boundaries = []
    start_pos = 0
    for sentence in sentence_list:
        # We use the actual position of the sentence in the text to get the exact boundaries
        start = text.index(sentence, start_pos)
        end = start + len(sentence)
        sentence_boundaries.append((start, end))
        start_pos = end  # we don't add +1 here because we're using the actual positions
        
    # Determine which entities fall within each sentence boundary
    entities_in_sentences = {}
    for ann in annotations:
        if ann["type"] == "labels":
            char_start = ann["value"]["start"]
            char_end = ann["value"]["end"]
            
            # Check in which sentence boundary the entity falls
            for idx, (start, end) in enumerate(sentence_boundaries):
                if start <= char_start < end and start < char_end <= end:
                    if idx not in entities_in_sentences:
                        entities_in_sentences[idx] = []
                    entities_in_sentences[idx].append(ann)
                    break  # once found, break the loop
    
    return entities_in_sentences



def calculate_tok_span(text, char_span):
    """Calculate the token span based on char span."""
    # Adjust char_span to remove leading and trailing spaces
    span_text = text[char_span[0]:char_span[1]]
    trimmed_span_text = span_text.strip()
    start_adjustment = span_text.index(trimmed_span_text)
    end_adjustment = len(span_text) - len(trimmed_span_text) - start_adjustment
    char_span = (char_span[0] + start_adjustment, char_span[1] - end_adjustment)
    encoding = tokenizer.encode_plus(text, return_offsets_mapping=True, add_special_tokens=True)
    offset_mapping = encoding["offset_mapping"]
    start_token, end_token = None, None
    for idx, (start, end) in enumerate(offset_mapping):
        # Check for the start of the entity
        if start_token is None and start <= char_span[0] < end:
            start_token = idx
        # Check for the end of the entity
        if start < char_span[1] <= end:
            end_token = idx + 1  # +1 to make it exclusive
            break
    # This handles cases where only a part of the token is annotated
    if end_token is None:
        end_token = start_token + 1
    return [start_token-1, end_token-1]


def match_relations_to_sentences(entities_in_sentences, annotations):
    """Match relations to entities in the same sentence."""
    
    # Create an ID to entity mapping
    id_to_entity = {}
    for sentence_idx, entities in entities_in_sentences.items():
        for entity in entities:
            id_to_entity[entity["id"]] = entity
    
    # Find valid relations
    sentence_relations = {}
    for ann in annotations:
        if ann["type"] == "relation":
            from_id = ann["from_id"]
            to_id = ann["to_id"]
            
            # Check if both entities are in the same sentence
            from_sentence = None
            to_sentence = None
            for sentence_idx, entities in entities_in_sentences.items():
                if from_id in [entity["id"] for entity in entities]:
                    from_sentence = sentence_idx
                if to_id in [entity["id"] for entity in entities]:
                    to_sentence = sentence_idx
            
            if from_sentence is not None and to_sentence is not None and from_sentence == to_sentence:
                if from_sentence not in sentence_relations:
                    sentence_relations[from_sentence] = []
                relation = {
                    "from_id": from_id,
                    "to_id": to_id,
                    "labels": ann["labels"],
                    "type": ann["type"],
                    "direction": ann["direction"]
                }
                sentence_relations[from_sentence].append(relation)
    
    return sentence_relations

def get_sentence_starts_in_text(text, sentence_list):
    """Determine the starting position of each sentence in the text."""
    
    # Backup of the original text
    text_backup = text[:]
    
    # Calculate the start character positions for each sentence
    sentence_starts = []
    
    for sentence in sentence_list:
        # Find the sentence in the backup text
        start = text_backup.index(sentence)
        sentence_starts.append(start)
        
        # Replace the found sentence in the backup with special characters to avoid finding it again
        text_backup = text_backup[:start] + '#' * len(sentence) + text_backup[start + len(sentence):]
    
    return sentence_starts


def replace_special_characters(text):
    mapping_dict = {
    "\u202f": ' ',
    "\u03b2": '%',
    '\u03b1': 'α'
    }
    for k, v in mapping_dict.items():
        text = text.replace(k, v)
    return text

# Load the data
with open("utils/datasets/test_325/project-31-at-2024-03-25-02-50-8463b5e9.json", "r") as file:
    data = json.load(file)


result_data = []

for paper_idx,entry in enumerate(data):
    text = entry["data"]["text"]
    annotations = entry["annotations"][0]["result"]
    '''将长文本划分为短文本列表'''
    sentence_list = sent_tokenize(text)
    '''计算每个短文本的长度'''
    sentence_list_len = [len(i) for i in sentence_list]
    '''计算那些实体位于同一个短文本之中'''
    if paper_idx == 18:
        print()
    result = get_entities_in_same_sentence(text, sentence_list, annotations)
    '''计算那些关系存在于同一个短文本之中，跨越的关系直接删除'''
    # try:
    relations_in_sentences = match_relations_to_sentences(result, annotations)
    # except Exception as E:
        # relations_in_sentences = match_relations_to_sentences(result, annotations)
        # print()
    for sentence_idx in list(result.keys()):
        '''短文本'''
        short_sentence = sentence_list[sentence_idx]
        if "\\u03b2" in short_sentence:
            print
        '''计算之前文本的长度'''
        # 找到每个句子在原始文本中的开始位置
        # sentence_starts = [text.index(sentence) for sentence in sentence_list]
        sentence_starts = get_sentence_starts_in_text(text, sentence_list)
        previous_length = sentence_starts[sentence_idx]
        '''计算实体的位置'''
        entity_list = []
        hash_id2entity = {}
        for i in result[sentence_idx]:
            '''每一个实体的位置更新'''
            entity_start = i['value']['start']-previous_length
            entity_end = i['value']['end']-previous_length
            entity_text = i['value']['text']
            id = i['id']
            label = i['value']['labels']
            # hash_id2entity = {id:entity_text}
            

            char_span = [entity_start,entity_end]
            try:
                calculate_tok_span(short_sentence,(entity_start,entity_end))
            except Exception as E:
                print()
            
            tok_span = calculate_tok_span(short_sentence,(entity_start,entity_end))
            hash_id2entity[id] = [entity_text,char_span,tok_span]
            entity_list.append({
			"text": short_sentence[entity_start:entity_end],
			"type": label,
            'id': id,
			"char_span": char_span,
			"tok_span": tok_span
		})
        '''更新实体关系'''
        relation_list = []
        '''某些句子只有实体，并没有对应的实体关系'''
        if sentence_idx in relations_in_sentences.keys():
            for i in relations_in_sentences[sentence_idx]:
                subject_entity_id = i['from_id']
                object_entity_id = i['to_id']
                relation_list.append({
                "subject": hash_id2entity[subject_entity_id][0],
                "object": hash_id2entity[object_entity_id][0],
                "subj_char_span": hash_id2entity[subject_entity_id][1],
                "obj_char_span": hash_id2entity[object_entity_id][1],
                "predicate": ''.join(i['labels']),
                "subj_tok_span": hash_id2entity[subject_entity_id][2],
                "obj_tok_span": hash_id2entity[object_entity_id][2]
                })
        if len(entity_list) > 0:
            result_data.append({
                "text": short_sentence,
                "id": str(paper_idx)+str(sentence_idx),
                'entity_list':entity_list,
                'relation_list':relation_list
            })


# Save the adjusted data
with open("utils/datasets/test_325/test.json", "w") as file:
    json.dump(result_data, file,indent=4)

