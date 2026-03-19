from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

def encode(text):
    return tokenizer(text, return_tensors="pt")

def decode(tokens):
    return tokenizer.decode(tokens)