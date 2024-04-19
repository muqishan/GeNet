import torch
from transformers import BertTokenizerFast

model_path = "triton_inference_server/model/unirel_model_full.pt"
model = torch.load(model_path)
model.eval()

tokenizer = BertTokenizerFast.from_pretrained("pubmedbert")
example_text = "Inhibition of DUXAP10 in HCC HepG2 cells could attenuate the EMT and cell proliferation and invasion."  
example_input = tokenizer.encode_plus(example_text, add_special_tokens=True, max_length=150, padding="max_length", truncation=True)

class TorchScriptModel(torch.nn.Module):
    def __init__(self, model):
        super(TorchScriptModel, self).__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        return tuple(outputs.values())

input_ids = torch.tensor(example_input["input_ids"]).unsqueeze(0)
attention_mask = torch.tensor(example_input["attention_mask"]).unsqueeze(0)
token_type_ids = torch.tensor(example_input["token_type_ids"]).unsqueeze(0)


scripted_model = TorchScriptModel(model)
scripted_model.eval()
traced_model = torch.jit.trace(scripted_model, (input_ids, attention_mask, token_type_ids))
traced_model.save("triton_inference_server/model/traced_model.pt")
