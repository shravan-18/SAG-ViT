import torch
from safetensors.torch import load_file

# Path to your safetensors file
safetensors_path = "model.safetensors"
bin_path = "pytorch_model.bin"

# Load model weights from safetensors
state_dict = load_file(safetensors_path)

# Save as pytorch_model.bin
torch.save(state_dict, bin_path)

print(f"Converted {safetensors_path} to {bin_path}.")
