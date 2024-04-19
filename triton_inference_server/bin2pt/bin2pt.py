import torch
from model_transformers import UniRelModel
# 加载模型
model_path = "output/my_data1/checkpoint-final"
model = UniRelModel.from_pretrained(model_path)


torch.save(model, 'triton_inference_server/model/unirel_model_full.pt')
