# this script uses load_model.py to load and inspect the model checkpoint
# it prints out the type of the checkpoint and its keys to verify successful loading

import torch

path = "trash_sorting_finetuned_model.pth"

# Load the checkpoint
checkpoint = torch.load(path, map_location="cpu")

print(type(checkpoint))
print(checkpoint.keys() if hasattr(checkpoint, "keys") else checkpoint)
