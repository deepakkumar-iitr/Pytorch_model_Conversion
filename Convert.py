# Converting a Hugging face model into the tflite 
# You can convert a .safetensors model to .tflite, but not directly. Here's a step-by-step overview of how it’s typically done: 
# Step 1: Load the .safetensors in PyTorch

from safetensors.torch import load_file
import torch

# Define model
model = YourModelDefinition()
model.load_state_dict(load_file("model.safetensors"))
model.eval()

# Step 2: Convert to ONNX
dummy_input = torch.randn(1, input_size)  # Adjust input size
torch.onnx.export(model, dummy_input, "model.onnx", opset_version=11)

# Step 3: Convert ONNX → TensorFlow
#  pip install onnx-tf
from onnx_tf.backend import prepare
import onnx

onnx_model = onnx.load("model.onnx")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("model_tf")
# This creates a SavedModel folder.

# Step 4: Convert TensorFlow SavedModel → TFLite
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("model_tf")
tflite_model = converter.convert()

# Save the model
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
