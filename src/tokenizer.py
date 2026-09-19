text = "The cat is running"
tokens = text.split()
# print(tokens)
vocab = {
    token: i for i,token in enumerate(tokens)
}
# print(vocab)
token_ids = [vocab[token] for token in tokens]
print(token_ids)

# new_text = "The dog is running"
# new_tokens = new_text.split()
# new_token_ids = [vocab[token] for token in new_tokens]
# print(new_token_ids)

words = ["run", "running", "runner", "walk", "walking"]
for word in words:
    if word.startswith("run"):
        print(word, "->", "run")
    elif word.startswith("walk"):
        print(word, "->", "walk")