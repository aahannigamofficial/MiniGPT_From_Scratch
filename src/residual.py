import torch

x = torch.randn(4, 8)

attention_output = torch.randn(4, 8)

output = x + attention_output

print("Original:", x.shape)
print("Attention:", attention_output.shape)
print("After residual:", output.shape)