class TinyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(tokens)


def test_token_aware_chunks_preserve_order():
    from src.chunking import token_aware_chunks

    chunks = token_aware_chunks("one two three four five", TinyTokenizer(), max_tokens=3, overlap=1)
    assert [chunk.text for chunk in chunks] == ["one two three", "three four five"]