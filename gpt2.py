
import numpy as np
from safetensors import safe_open

def layernorm(x, weight, bias, eps=1e-5): #eps is added to std in case std is 0
    x = (x - np.mean(x, axis=1, keepdims=True)) / (np.std(x, axis=1, keepdims=True) + eps) #keepdims so subtracting broadcasts correctly, axis=1 so mean and std is per row(per word)
    return (x * weight + bias)

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True) # for stability, so exp for a high number dosn't go to inifinity
    x_exp = np.exp(x)
    return x_exp / np.sum(x_exp, axis=1, keepdims=True)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

def feedforward(x, fc_weight, fc_bias, projected_weight, projected_bias):
    x = x @ fc_weight + fc_bias
    x = gelu(x)
    x = x @ projected_weight + projected_bias
    return x

def attention(x, attn_weight, attn_bias, projected_weight, projected_bias, kv_cache=None):
    print(f"x's shape: {x.shape}, attn_weight's shape: {attn_weight.shape}")
    W_qkv = x @ attn_weight   + attn_bias
    Wq, Wk, Wv = np.split(W_qkv, 3, axis=1) #each with shape (1, 768)
    if kv_cache is not None:
        k_cache, v_cache = kv_cache
        Wk = np.concatenate([k_cache, Wk], axis=0)
        Wv = np.concatenate([v_cache, Wv], axis=0) #axis = 0 means adding row(s)
    new_kv_cache = Wk, Wv
    cache_length = Wk.shape[0] - Wq.shape[0]
    mask = np.tril(np.ones((Wq.shape[0], Wk.shape[0])), k=cache_length) # without K, token will be blind to all past tokens except first token, as k=0 is default
    mask = (1 - mask) * -1e10
    B = 0
    E = 64
    head_outputs = []
    for i in range(12):
        Q = Wq[:, B:E] 
        K = Wk[:, B:E]
        V = Wv[:, B:E]
        scores = Q @ K.T
        scores = (scores + mask) / np.sqrt(Q.shape[1]) # we divide by  sqrt Q.shape[1] to avoid disproportionally large numbers
        weights = softmax(scores)
        output = weights @ V
        head_outputs.append(output)
        B += 64
        E += 64
    final_output = np.concatenate(head_outputs, axis=1)
    return (final_output @ projected_weight + projected_bias), new_kv_cache

def transformer_layer(x, layer,  kv_cache=None):
    attn, new_kv_cache = attention(layernorm(x, layer['ln1_weight'], layer['ln1_bias']), layer['attn_weight'], layer['attn_bias'], layer['attn_proj_weight'], layer['attn_proj_bias'], kv_cache=kv_cache)
    x = x + attn
    x = x + feedforward(layernorm(x, layer['ln2_weight'], layer['ln2_bias']), layer['fc_weight'], layer['fc_bias'], layer['fc_proj_weight'], layer['fc_proj_bias'])
    return x, new_kv_cache
def gpt_forward(token_ids, wte, wpe, layers, ln_f_weight, ln_f_bias, kv_cache=None):
    layers_caches = []
    if kv_cache is not None:
        start_position = kv_cache[0][0].shape[0] #to get N, could've indexed any of the layers they all have the same N
        positions = np.array([start_position]) #putting start_position in an array so wpe matches wte's shape
    else: 
        positions = np.arange(len(token_ids))
    x = wte[token_ids] + wpe[positions]
    for i, layer in enumerate(layers):
       layer_cache = kv_cache[i] if kv_cache is not None else None 
       x, layer_new_cache = transformer_layer(x, layer, layer_cache)
       layers_caches.append(layer_new_cache)
    x = layernorm(x, ln_f_weight, ln_f_bias)
    raw_logits = wte @ x[-1]
    return raw_logits, layers_caches

def load_weights(model_path):
    layers = []
    with safe_open(model_path, framework="numpy") as f:
        for x in f.keys():
            if x == "ln_f.weight":
                ln_f_weight = f.get_tensor(x)
            elif x == "ln_f.bias":
                ln_f_bias = f.get_tensor(x)
            elif x == "wte.weight":
                wte = f.get_tensor(x)
            elif x == "wpe.weight": 
                wpe = f.get_tensor(x)
        for i in range(12):
            layers.append({
                'ln1_weight': f.get_tensor(f'h.{i}.ln_1.weight'),
                'ln1_bias': f.get_tensor(f'h.{i}.ln_1.bias'),
                'ln2_weight': f.get_tensor(f'h.{i}.ln_2.weight'),
                'ln2_bias': f.get_tensor(f'h.{i}.ln_2.bias'),
                'attn_weight': f.get_tensor(f'h.{i}.attn.c_attn.weight'),
                'attn_bias': f.get_tensor(f'h.{i}.attn.c_attn.bias'),
                'attn_proj_bias': f.get_tensor(f'h.{i}.attn.c_proj.bias'),
                'attn_proj_weight': f.get_tensor(f'h.{i}.attn.c_proj.weight'),
                'fc_weight': f.get_tensor(f'h.{i}.mlp.c_fc.weight'), 
                'fc_bias': f.get_tensor(f'h.{i}.mlp.c_fc.bias'),
                'fc_proj_bias': f.get_tensor(f'h.{i}.mlp.c_proj.bias'),
                'fc_proj_weight': f.get_tensor(f'h.{i}.mlp.c_proj.weight'),
            })
        return wte, wpe, layers, ln_f_weight, ln_f_bias
            
 

