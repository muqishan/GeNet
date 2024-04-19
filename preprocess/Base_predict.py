import os
import numpy as np
import torch
from transformers import (BertTokenizerFast)
import dataprocess.rel2text
from model.model_transformers import  UniRelModel
from dataprocess.data_extractor import *
from dataprocess.data_metric import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class UniRel:
    def __init__(self, model_path, max_length=128) -> None:
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = UniRelModel.from_pretrained(model_path).to(self.device)
        added_token = [f"[unused{i}]" for i in range(1, 17)]
        self.tokenizer = BertTokenizerFast.from_pretrained(
            "pubmedbert", additional_special_tokens=added_token, do_basic_tokenize=False)
        self.max_length = max_length
        self.max_length = max_length
        self._get_pred_str()
        
    
    def _get_pred_str(self):
        self.pred2text = None
        self.pred2text = dataprocess.rel2text.my_dataset
        self.pred_str = ""
        self.max_label_len = 1
        self.pred2idx = {}
        idx = 0
        for k in self.pred2text:
            self.pred2idx[k] = idx
            self.pred_str += self.pred2text[k] + " "
            idx += 1
        self.num_rels = len(self.pred2text.keys())
        self.idx2pred = {value: key for key, value in self.pred2idx.items()}
        self.pred_str = self.pred_str[:-1]
        self.pred_inputs = self.tokenizer.encode_plus(self.pred_str,
                                                 add_special_tokens=False)
    
    def _data_process(self, text):
        # text could be a list of sentences or a single sentence
        if isinstance(text, str):
            text = [text]
        inputs = self.tokenizer.batch_encode_plus(text, max_length=self.max_length, padding="max_length", truncation=True,return_offsets_mapping=True)
        batched_input_ids = []
        batched_attention_mask = []
        batched_token_type_ids = []
        for b_input_ids, b_attention_mask, b_token_type_ids in zip(inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]):
            input_ids = b_input_ids + self.pred_inputs["input_ids"]
            sep_idx = b_input_ids.index(self.tokenizer.sep_token_id)
            input_ids[sep_idx] = self.tokenizer.pad_token_id
            attention_mask = b_attention_mask + [1]*self.num_rels
            attention_mask[sep_idx] = 0
            token_type_ids = b_token_type_ids + [1]*self.num_rels
            batched_input_ids.append(input_ids)
            batched_attention_mask.append(attention_mask)
            batched_token_type_ids.append(token_type_ids)
        return batched_input_ids, batched_attention_mask, batched_token_type_ids,inputs["offset_mapping"][0]
    

    def _get_e2r(self, e2r_pred):
        """
        Extract entity-relation (subject-relation) and entity-entity interactions from given Attention Matrix.
        Only Extract the upper-right triangle, so should input transpose of the original
        Attention Matrix to extract relation-entity (relation-object) interactions.
        """
        token_len = self.max_length-2
        e2r = {}
        tok_tok = set()
        e_va = np.where(e2r_pred.cpu() == 1)
        for h, r in zip(e_va[0], e_va[1]):
            h = int(h)
            r = int(r)            
            if h == 0 or r == 0 or r == token_len+1 or h > token_len:
                continue
            # Entity-Entity
            if r < token_len+1:
                tok_tok.add((h,r))
            # Entity-Relation
            else:
                r = int(r-token_len-2)
                if h not in e2r:
                    e2r[h] = []
                e2r[h].append(r)
        return e2r, tok_tok
    
    def _get_span_att(self, span_pred):
        token_len = self.max_length-2
        span_va = np.where(span_pred.cpu() == 1)
        t2_span = dict()
        h2_span = dict()
        for s, e in zip(span_va[0], span_va[1]):
            # if s > token_len or e > token_len or s == 0 or e == 0:
            if s > token_len or e > token_len:
                continue
            if e < s:
                continue
            if e not in t2_span:
                t2_span[e] = []
            if s not in h2_span:
                h2_span[s] = []
            s = int(s)
            e = int(e)
            t2_span[e].append((s,e))
            h2_span[s].append((s,e))
        return  h2_span, t2_span

    def _extractor(self, outputs, input_ids_list,offset_mapping):
        # preds_list = []
        for head_pred, tail_pred, span_pred, input_ids in zip(outputs["head_preds"], outputs["tail_preds"], outputs["span_preds"], input_ids_list):
            pred_spo_text = set()
            s_h2r, s2s = self._get_e2r(head_pred)
            s_t2r, _ = self._get_e2r(head_pred.T)
            e_h2r, e2e = self._get_e2r(tail_pred)
            e_t2r, _ = self._get_e2r(tail_pred.T)
            start2span, end2span = self._get_span_att(span_pred)
            for l, r in e2e:
                if l not in e_h2r or r not in e_t2r:
                    continue
                if l not in end2span or r not in end2span:
                    continue
                l_spans, r_spans = end2span[l], end2span[r]
                for l_span in l_spans:
                    for r_span in r_spans:
                        l_s, r_s = l_span[0], r_span[0]
                        if (l_s, r_s) not in s2s:
                            continue
                        if l_s not in s_h2r or r_s not in s_t2r:
                            continue
                        common_rels = set(s_h2r[l_s])& set(s_t2r[r_s]) & set(e_h2r[l]) & set(e_t2r[r])
                        # l_span_new = (l_span[0]+1, l_span[1])
                        # r_span_new = (r_span[0]+1, r_span[1])
                        l_span_new = (l_span[0], l_span[1])
                        r_span_new = (r_span[0], r_span[1])
                        for rel in common_rels:
                            object_entity = (l_span_new[0],l_span_new[1])
                            subject_entity = (r_span_new[0],r_span_new[1])
                            pred_spo_text.add(( # idx,                                          
                                            (offset_mapping[object_entity[0]][0],offset_mapping[object_entity[1]][1]),
                                            self.idx2pred[rel],
                                            (offset_mapping[subject_entity[0]][0],offset_mapping[subject_entity[1]][1])
                                             ))

            # preds_list.append(list(pred_spo_text))
        return list(pred_spo_text)

    def predict(self, text):
        input_ids, attention_mask, token_type_ids,offset_mapping = self._data_process(text)
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids).to(self.device)
            attention_mask = torch.tensor(attention_mask).to(self.device)
            token_type_ids = torch.tensor(token_type_ids).to(self.device)
        else:
            input_ids = torch.tensor(input_ids).unsqueeze(0).to(self.device)
            attention_mask = torch.tensor(attention_mask).unsqueeze(0).to(self.device)
            token_type_ids = torch.tensor(token_type_ids).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask, token_type_ids)
            results = self._extractor(outputs, input_ids,offset_mapping)
        return results  