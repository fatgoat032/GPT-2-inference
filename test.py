from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

input_ids = tokenizer.encode("Scientists studying the human brain discovered that", return_tensors='pt')
output = model.generate(input_ids, max_length=50, temperature=0.7, top_k=40, do_sample=True)
print(tokenizer.decode(output[0]))