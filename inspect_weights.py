from safetensors import safe_open

with safe_open("model.safetensors", framework="numpy") as f:
    for x in f.keys():
        tensor = f.get_tensor(x)
        print(f"tensor: {x}, tensor shape: {tensor.shape}")