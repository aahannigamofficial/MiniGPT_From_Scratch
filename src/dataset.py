import torch


token_ids = torch.tensor([
    2, 5, 8, 11, 14, 7, 3, 9
])

sequence_length = 4


inputs = []
targets = []


for i in range(len(token_ids) - sequence_length):

    input_chunk = token_ids[
        i:i + sequence_length
    ]

    target_chunk = token_ids[
        i + 1:i + sequence_length + 1
    ]

    inputs.append(input_chunk)
    targets.append(target_chunk)


inputs = torch.stack(inputs)
targets = torch.stack(targets)


print("Inputs:")
print(inputs)

print("\nTargets:")
print(targets)

print("\nInput shape:")
print(inputs.shape)

print("\nTarget shape:")
print(targets.shape)