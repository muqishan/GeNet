import torch
import onnx
from onnx_tf.backend import prepare


model_path = "GeNet/triton_inference_server/models/relation/1/model.pt"
model = torch.load(model_path)


dummy_input = torch.randn(1, 128)  
onnx_path = "GeNet/triton_inference_server/models/relation/1/model.onnx"
torch.onnx.export(model, dummy_input, onnx_path, verbose=True)

onnx_model = onnx.load(onnx_path)
tf_rep = prepare(onnx_model)

input_names = tf_rep.inputs
print("Model Input Names:", input_names)
