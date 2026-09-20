import torch
from tokenizer import Tokenizer
from model import MiniGPT


# ============================================================
# Load trained checkpoint
# ============================================================

checkpoint = torch.load(
    "checkpoints/minigpt.pth",
    map_location="cpu"
)


# ============================================================
# Load model configuration
# ============================================================

vocab_size = checkpoint["vocab_size"]
max_sequence_length = checkpoint["max_sequence_length"]
embedding_dim = checkpoint["embedding_dim"]
num_heads = checkpoint["num_heads"]
hidden_dim = checkpoint["hidden_dim"]
num_blocks = checkpoint["num_blocks"]

token_to_id = checkpoint["token_to_id"]
id_to_token = checkpoint["id_to_token"]


# PyTorch may store integer dictionary keys as integers,
# but we make sure they are integers here.
id_to_token = {
    int(key): value
    for key, value in id_to_token.items()
}


# ============================================================
# Create model
# ============================================================

model = MiniGPT(
    vocab_size=vocab_size,
    max_sequence_length=max_sequence_length,
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    hidden_dim=hidden_dim,
    num_blocks=num_blocks
)


# Load trained weights
model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ============================================================
# Encoding and decoding
# ============================================================

with open(
    "data/corpus.txt",
    "r",
    encoding="utf-8"
) as file:
    corpus_text = file.read()

tokenizer = Tokenizer(corpus_text)

def encode(text):
    """
    Convert text into token IDs.
    """
    return tokenizer.encode(text)


def decode(token_ids):
    return tokenizer.decode(token_ids)
    """
    Convert token IDs back into text.
    """

# ============================================================
# Generate text
# ============================================================

def generate(
    prompt,
    max_new_tokens=10,
    temperature=1.0,
    top_k=None
):

    # Convert prompt to token IDs
    token_ids = encode(prompt)

    # Convert to tensor
    input_ids = torch.tensor(
        [token_ids],
        dtype=torch.long
    )

    for _ in range(max_new_tokens):

        # Keep only the most recent tokens
        # if the sequence becomes too long.
        input_ids = input_ids[
            :, -max_sequence_length:
        ]

        # No gradient calculation needed
        # during inference.
        with torch.no_grad():

            logits = model(input_ids)

        # We only care about the prediction
        # for the LAST token.
        next_token_logits = logits[:, -1, :]

        # ====================================================
        # Temperature
        # ====================================================

        next_token_logits = (
            next_token_logits / temperature
        )

        # ====================================================
        # Top-k sampling
        # ====================================================

        if top_k is not None:

            values, _ = torch.topk(
                next_token_logits,
                min(top_k, vocab_size)
            )

            minimum_value = values[:, -1].unsqueeze(-1)

            next_token_logits = torch.where(
                next_token_logits < minimum_value,
                torch.tensor(
                    float("-inf"),
                    device=next_token_logits.device
                ),
                next_token_logits
            )

        # ====================================================
        # Convert logits to probabilities
        # ====================================================

        probabilities = torch.softmax(
            next_token_logits,
            dim=-1
        )

        # ====================================================
        # Sample next token
        # ====================================================

    next_token = torch.argmax(
        next_token_logits,
        dim=-1,
        keepdim=True
            )

        # ====================================================
        # Add new token to sequence
        # ====================================================

    input_ids = torch.cat(
            [input_ids, next_token],
            dim=1
        )

    # Convert generated IDs back into text
    return decode(
        input_ids[0].tolist()
    )


# ============================================================
# Test generation
# ============================================================

prompt = "The animals are"

generated_text = generate(
    prompt,
    max_new_tokens=10,
    temperature=1.0,
    top_k=5
)

print("\nPrompt:")
print(prompt)

print("\nGenerated:")
print(generated_text)