import requests

url = "https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors"
response = requests.get(url)
with open("model.safetensors", "wb") as f:
    f.write(response.content)