# MiniGPT_From_Scratch
# MiniGPT From Scratch

A beginner-friendly implementation of a **small GPT-style Large Language Model from scratch using PyTorch**.

The goal of this project is not to build a production-scale ChatGPT clone. Instead, it is to understand **how an LLM actually works internally** by implementing and studying each major component ourselves.

The project follows the journey of text through the model:

```text
Text
 ↓
Tokenizer
 ↓
Token IDs
 ↓
Token Embeddings
 +
Position Embeddings
 ↓
Transformer Blocks
 ├── LayerNorm
 ├── Multi-Head Self-Attention
 ├── Residual Connection
 ├── LayerNorm
 ├── Feed Forward Network
 └── Residual Connection
 ↓
Final LayerNorm
 ↓
LM Head
 ↓
Logits
 ↓
Next-Token Prediction
 ↓
Cross-Entropy Loss
 ↓
Backpropagation
 ↓
Adam Optimizer
 ↓
Updated Parameters
```

---

## Project Goal

The main objective is to understand the internal architecture of a GPT-style language model rather than simply using an existing model or library.

By completing this project, I aim to understand:

* How text becomes numbers
* How tokenization works
* How embeddings represent tokens
* Why positional information is required
* How self-attention works
* What Query, Key, and Value mean
* Why attention uses `QKᵀ`
* Why we divide by `√d`
* How softmax converts scores into probabilities
* How causal masking prevents future-token information leakage
* Why multiple attention heads are used
* What a Feed-Forward Network does
* Why residual connections are important
* Why LayerNorm is used
* How Transformer blocks are stacked
* How the model produces logits
* How next-token prediction works
* How Cross-Entropy Loss is calculated
* How backpropagation updates the model
* How Adam optimizes the parameters
* How autoregressive text generation works
* How temperature and top-k sampling affect generation

---

# Architecture

The model can be viewed as several major stages.

## 1. Tokenization

Text is converted into individual tokens.

Example:

```text
"The cat is sitting."
        ↓
["The", "cat", "is", "sitting", "."]
```

Each token is then mapped to an integer ID:

```text
["The", "cat", "is", "sitting", "."]
        ↓
[ID, ID, ID, ID, ID]
```

The project uses a simple tokenizer based on:

```python
re.findall(r"\w+|[^\w\s]", text)
```

The tokenizer maintains:

```text
token_to_id
id_to_token
```

The project also explores concepts such as:

* Vocabulary
* `<UNK>`
* Word-level tokenization
* Subword tokenization
* BPE-style tokenization

---

# 2. Token Embeddings

Neural networks do not directly understand token IDs.

Instead, each token ID is mapped to a learned vector.

```text
Token ID
   ↓
Embedding Layer
   ↓
Vector
```

The implementation uses:

```python
nn.Embedding(vocab_size, embedding_dim)
```

Conceptually:

```text
"cat"
 ↓
token ID
 ↓
embedding vector
```

The embedding vector is learned during training.

---

# 3. Positional Embeddings

Self-attention does not inherently provide information about the order of tokens.

For example:

```text
The dog chased the cat
```

and:

```text
The cat chased the dog
```

contain the same words but have different meanings.

Therefore, the model combines:

```text
Token Embedding
+
Position Embedding
```

The project implements this as:

```python
X = token_vectors + position_vectors
```

Important tensor shape:

```text
Token IDs:
[B, T]

Embeddings:
[B, T, D]
```

Where:

```text
B = Batch Size
T = Sequence Length
D = Embedding Dimension
```

---

# 4. Self-Attention

Self-attention allows each token to determine how much it should pay attention to other tokens.

For example:

```text
The cat sat
```

When processing `"sat"`, the model can use information from `"The"` and `"cat"`.

The attention mechanism creates three representations:

```text
Q = Query
K = Key
V = Value
```

Implemented using learned projections:

```python
Q = self.W_Q(X)
K = self.W_K(X)
V = self.W_V(X)
```

The attention scores are calculated using:

```text
QKᵀ
```

Then scaled:

```text
QKᵀ / √d
```

Then converted into probabilities using softmax:

```text
softmax(QKᵀ / √d)
```

Finally, these probabilities are used to calculate a weighted combination of the values:

```text
Attention(Q,K,V)
=
softmax(QKᵀ / √d)V
```

---

# 5. Causal Attention

GPT is an autoregressive model.

When predicting a token, it must not be allowed to see future tokens.

For example:

```text
The cat is
```

When predicting the next token, the model can use:

```text
The
cat
is
```

but not:

```text
sitting
```

if `"sitting"` is the future target.

A causal attention mask therefore has a triangular structure:

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

Forbidden positions are assigned:

```text
-inf
```

before softmax.

Therefore:

```text
-inf
 ↓
softmax
 ↓
0 probability
```

This prevents the model from cheating during training.

---

# 6. Multi-Head Attention

Instead of using one attention mechanism, the Transformer uses multiple attention heads.

The input:

```text
[B, T, D]
```

is reshaped into:

```text
[B, T, H, head_dim]
```

and then:

```text
[B, H, T, head_dim]
```

Where:

```text
H = Number of attention heads
```

Each head can potentially learn different relationships between tokens.

The outputs of the attention heads are eventually combined back together.

---

# 7. Feed-Forward Network

After attention, each token representation passes through a Feed-Forward Network.

The project's FFN follows:

```text
Embedding Dimension
        ↓
Hidden Dimension
        ↓
GELU
        ↓
Embedding Dimension
```

Example:

```text
8
↓
32
↓
GELU
↓
8
```

Implemented using:

```python
nn.Linear(embedding_dim, hidden_dim)
nn.GELU()
nn.Linear(hidden_dim, embedding_dim)
```

An important distinction:

> **Attention mixes information between tokens.**

> **The Feed-Forward Network transforms each token's representation independently.**

---

# 8. Residual Connections

Residual connections allow the original representation to flow around a sub-layer.

The project uses:

```python
X = X + attention_output
```

and:

```python
X = X + feed_forward_output
```

Residual connections help preserve information and improve gradient flow through deep networks.

---

# 9. LayerNorm

Layer Normalization is used inside the Transformer block to stabilize the activations.

The project's Transformer block follows a **Pre-LN style** structure:

```text
X
 ↓
LayerNorm
 ↓
Attention
 ↓
Residual
 ↓
LayerNorm
 ↓
Feed Forward
 ↓
Residual
 ↓
Output
```

---

# 10. Transformer Block

All of the previous components are combined into one Transformer block.

```text
        X
        │
   LayerNorm
        │
   Attention
        │
      + X
        │
   LayerNorm
        │
       FFN
        │
      + X
        │
      Output
```

The model can then stack multiple Transformer blocks:

```text
Input
 ↓
Block 1
 ↓
Block 2
 ↓
Block 3
 ↓
Output
```

The project uses `nn.ModuleList` to maintain the collection of Transformer blocks.

---

# 11. Language Model Head

After the Transformer blocks, the representation is passed through a final LayerNorm and then an LM Head.

The LM Head converts:

```text
[B, T, D]
```

into:

```text
[B, T, V]
```

where:

```text
V = Vocabulary Size
```

The implementation uses:

```python
self.lm_head = nn.Linear(
    embedding_dim,
    vocab_size
)
```

The output values are called **logits**.

A logit is the raw score the model assigns to a possible next token before converting the scores into probabilities.

---

# 12. Next-Token Prediction

The model is trained to predict the next token.

For example:

```text
Input:
The cat is

Target:
cat is sitting
```

More generally:

```text
X = [t1, t2, t3, t4]

Y = [t2, t3, t4, t5]
```

Therefore, every position in the input is used to predict the token immediately following it.

This is the fundamental training objective of the MiniGPT.

---

# 13. Cross-Entropy Loss

The model produces logits:

```text
[B, T, V]
```

For `CrossEntropyLoss`, these are flattened into:

```text
[B*T, V]
```

while the target tokens become:

```text
[B*T]
```

The loss measures how well the model predicted the correct next token.

Conceptually:

```text
Model prediction
       ↓
Compare with
       ↓
Correct token
       ↓
Cross-Entropy Loss
```

---

# 14. Backpropagation

Training follows this process:

```text
Input
 ↓
Forward Pass
 ↓
Logits
 ↓
Loss
 ↓
loss.backward()
 ↓
Gradients
 ↓
Optimizer
 ↓
Updated Parameters
```

The call:

```python
loss.backward()
```

calculates gradients for the model's learnable parameters.

Parameters that can be updated include:

* Token embeddings
* Position embeddings
* Q/K/V projections
* Feed-forward weights
* LayerNorm parameters
* LM Head parameters

---

# 15. Adam Optimizer

The optimizer uses the gradients to update the model parameters.

The project uses the Adam optimization algorithm.

Conceptually:

```text
Gradients
   ↓
Adam
   ↓
Parameter Updates
```

After many training iterations, the parameters gradually change so that the model becomes better at predicting the next token.

---

# 16. Autoregressive Generation

Once the model has been trained, it can generate text one token at a time.

Suppose the prompt is:

```text
"The dog is playing with the"
```

The model predicts:

```text
"cat"
```

The prompt becomes:

```text
"The dog is playing with the cat"
```

The new sequence is fed back into the model.

The process repeats:

```text
Prompt
 ↓
Predict next token
 ↓
Append token
 ↓
Feed back into model
 ↓
Predict next token
 ↓
Append token
 ↓
Repeat
```

This is called **autoregressive generation**.

---

# 17. Temperature

Temperature changes how deterministic or random the token selection becomes.

Conceptually:

```text
logits / temperature
```

Lower temperature:

```text
More deterministic
```

Higher temperature:

```text
More random
```

Temperature therefore controls the randomness of generation.

---

# 18. Top-k Sampling

Instead of sampling from the entire vocabulary, top-k sampling keeps only the highest-scoring `k` tokens.

For example:

```text
Vocabulary = 37 tokens

Top-k = 5
```

The model keeps only the five highest-scoring tokens:

```text
Token A
Token B
Token C
Token D
Token E
```

and removes the others from consideration.

---

# MiniGPT vs Real GPT

This project is a **learning implementation**, not a production-scale LLM.

A real GPT-scale model differs substantially in areas such as:

* Dataset size
* Number of parameters
* Vocabulary/tokenizer
* Context length
* Embedding dimension
* Number of Transformer layers
* Number of attention heads
* Training compute
* Optimization
* Distributed training
* Pretraining
* Fine-tuning
* Instruction tuning
* Preference optimization
* Inference infrastructure

The purpose of this project is to understand the fundamental architecture rather than reproduce the scale of a commercial LLM.

---

# Full Architecture

```text
                    TEXT
                      │
                      ▼
                TOKENIZER
                      │
                      ▼
                 TOKEN IDs
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
   TOKEN EMBEDDING         POSITION EMBEDDING
          │                       │
          └───────────┬───────────┘
                      ▼
                  ADD THEM
                      │
                      ▼
             TRANSFORMER BLOCK
                      │
              ┌───────┴────────┐
              │                │
              ▼                │
          LayerNorm            │
              │                │
              ▼                │
      Multi-Head Attention     │
              │                │
              ▼                │
          Residual ◄───────────┘
              │
              ▼
          LayerNorm
              │
              ▼
             FFN
              │
              ▼
          Residual
              │
              ▼
         Next Block
              │
              ▼
       ... More Blocks ...
              │
              ▼
        Final LayerNorm
              │
              ▼
            LM Head
              │
              ▼
            LOGITS
              │
              ▼
       NEXT TOKEN PREDICTION
```

---

# Important Tensor Shape Cheat Sheet

| Component           | Shape                 |
| ------------------- | --------------------- |
| Token IDs           | `[B, T]`              |
| Token Embeddings    | `[B, T, D]`           |
| Position Embeddings | `[B, T, D]`           |
| Transformer Input   | `[B, T, D]`           |
| Q / K / V           | `[B, T, D]`           |
| Multi-head Q/K/V    | `[B, H, T, head_dim]` |
| Attention Scores    | `[B, H, T, T]`        |
| Attention Output    | `[B, T, D]`           |
| Transformer Output  | `[B, T, D]`           |
| Logits              | `[B, T, V]`           |
| Flattened Logits    | `[B*T, V]`            |
| Flattened Targets   | `[B*T]`               |

Where:

```text
B = Batch Size
T = Sequence Length
D = Embedding Dimension
H = Number of Attention Heads
V = Vocabulary Size
```

---

# Important Equations

### Scaled Dot-Product Attention

```text
Attention(Q,K,V)
=
softmax(QKᵀ / √d)V
```

### Input Representation

```text
X = Token Embedding + Position Embedding
```

### Residual Connection

```text
Output = X + SubLayer(X)
```

### Next-Token Training

```text
Input  = [t1, t2, t3, t4]

Target = [t2, t3, t4, t5]
```

### Generation

```text
Prompt
 ↓
Predict token
 ↓
Append token
 ↓
Repeat
```

---

# Project Learning Path

The project is designed to be understood in this order:

```text
1. LLM Fundamentals
2. Tokenization
3. Embeddings
4. Positional Embeddings
5. Q / K / V
6. Self-Attention
7. Causal Masking
8. Multi-Head Attention
9. Feed Forward Network
10. Residual Connections
11. LayerNorm
12. Transformer Block
13. Transformer Stack
14. LM Head
15. Next-Token Prediction
16. Cross Entropy
17. Backpropagation
18. Adam
19. Training
20. Generation
21. Temperature
22. Top-k
23. Full Architecture
24. Tensor Shape Cheat Sheet
25. Important Equations
```
