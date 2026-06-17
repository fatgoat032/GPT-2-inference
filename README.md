What this project does:
Loads GPT-2's pretrained weights from a .safetensors checkpoint
Implements token + positional embeddings, layernorm, multi-head causal self-attention, and the GELU feedforward block by hand
Uses KV caching so generation after the first token only processes one new token per step instead of recomputing the whole sequence
Sampling: Top-k and top-p (nucleus) sampling, plus a repetition penalty so the model doesn't get stuck in loops (which it loves to do on the 124M model).

How to run it
Install dependencies:
pip install numpy safetensors tiktoken requests
(Note for Linux users: If you get an "externally managed environment" error, you can bypass it by adding --break-system-packages to the end of that command. Or just use a virtualenv like a civilized human.)

Download the weights:
python download_weights.py

Run the model:
python generate.py

It'll ask for a starting text and a max token limit.
eg of a rompt:
The future of artificial intelligence is

Output:
The future of artificial intelligence is likely to....
Notes
It's slow. It's NumPy on the CPU. There are no system prompts or instructions here. You just give it the beginning of a sentence, and it tries to predict what comes next. Because it's the small 124M model, it can get pretty weird.
