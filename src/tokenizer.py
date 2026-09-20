import re

class Tokenizer:

    def __init__(self, text):
        # Split words and punctuation separately
        self.tokens = re.findall(r"\w+|[^\w\s]", text)

        # Special token for unknown words
        self.unk_token = "<UNK>"

        # Create vocabulary
        vocab = sorted(set(self.tokens))

        vocab.append(self.unk_token)

        self.token_to_id = {
            token: i
            for i, token in enumerate(vocab)
        }

        self.id_to_token = {
            i: token
            for token, i in self.token_to_id.items()
        }

    def encode(self, text):

        tokens = re.findall(
            r"\w+|[^\w\s]",
            text
        )

        return [
            self.token_to_id.get(
                token,
                self.token_to_id[self.unk_token]
            )
            for token in tokens
        ]

    def decode(self, token_ids):

        return " ".join(
            self.id_to_token[token_id]
            for token_id in token_ids
        )

    def __len__(self):

        return len(self.token_to_id)


# Test
if __name__ == "__main__":

    with open(
        "data/corpus.txt",
        "r",
        encoding="utf-8"
    ) as file:

        text = file.read()

    tokenizer = Tokenizer(text)

    encoded = tokenizer.encode(text)

    print("Vocabulary size:", len(tokenizer))
    print("First 20 token IDs:", encoded[:20])

    decoded = tokenizer.decode(encoded[:20])

    print("Decoded:")
    print(decoded)