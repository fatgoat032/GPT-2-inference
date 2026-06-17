
import tiktoken
import numpy as np
from gpt2 import gpt_forward, load_weights, softmax

def sample(logits, temperature=0.6, k=50, p=0.9):
    logits = logits / temperature
    top_tokens_indices = np.argsort(-logits)[:k]
    mask = np.full(len(logits), -np.inf)
    mask[top_tokens_indices] = logits[top_tokens_indices]
    logits = mask 
    probs = softmax(logits.reshape(1,-1)).flatten()
    p_sorted_indices = np.argsort(-probs)
    mask = np.full(len(probs), -np.inf)
    total_so_far = 0
    for index in p_sorted_indices:
        total_so_far += probs[index]
        mask[index] = logits[index]
        if total_so_far >= p:
            break
    logits = mask
    probs = softmax(logits.reshape(1, -1)).flatten()
    return np.random.choice(len(probs), p=probs)

def repetition_penalty(logits, generated, penalty=1.4):
    for token_index in set(generated): #use set as if a vaue appeared more than once its only penalized once
        if logits[token_index] > 0:
            logits[token_index] /= penalty
        elif logits[token_index] < 0:
            logits[token_index] *= penalty # this if check to appropriately scale logits based on its sign
    return logits

wte, wpe, layers, ln_f_weight, ln_f_bias = load_weights("model.safetensors")
enc = tiktoken.get_encoding("gpt2")
prompt = input(": ")
next_token = 0
try:
    max_tokens = input("Enter maximum number of tokens you want for this prompt: ")
except TypeError:
    print("Enter a valid number.")
    exit()
print("please wait for response.....")
print()
i = 0
token_ids = enc.encode(prompt)
response = list(token_ids)
kv_cache = None
logits , kv_cache = gpt_forward(token_ids, wte, wpe, layers, ln_f_weight, ln_f_bias, kv_cache=kv_cache)
logits = repetition_penalty(logits, response)
next_token = sample(logits)
response.append(next_token)
while i < int(max_tokens):
    if next_token == 50256:
        break
    print(f"token {i}")
    i += 1
    logits , kv_cache = gpt_forward(next_token, wte, wpe, layers, ln_f_weight, ln_f_bias, kv_cache=kv_cache)
    logits = repetition_penalty(logits, response)
    next_token = sample(logits)
    response.append(next_token)
text = enc.decode(response)
print(text)

