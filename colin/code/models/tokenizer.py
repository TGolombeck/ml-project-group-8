from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def _as_str_list(text):
    if isinstance(text, str):
        return text
    if hasattr(text, "tolist"):
        text = text.tolist()
    else:
        text = list(text)
    return ["" if t is None or (isinstance(t, float) and t != t) else str(t) for t in text]


def encode(text):
    batch = _as_str_list(text)
    if isinstance(batch, str):
        return tokenizer(batch, return_tensors="pt")
    return tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

def decode(tokens):
    return tokenizer.decode(tokens)