import torch
ckpt = torch.load("pretrained.pt", map_location="cpu")
print(len(ckpt["classes"]))
print(ckpt["classes"])