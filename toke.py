from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("modelul-vostru")

print(tokenizer.tokenize("acesta este <nou-token> test"))
print(tokenizer.encode("acesta este <nou-token> test"))