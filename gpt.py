#!/usr/bin/env python
# coding: utf-8

# In[229]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[230]:


import torch
import torch.nn as nn
from torch.nn import functional as F

# ============================================================
# Setup
# ============================================================

torch.manual_seed(1337)

device = "cuda" if torch.cuda.is_available() else "cpu"

print(torch.cuda.is_available())
print(torch.cuda.device_count())

if torch.cuda.is_available():
    print(torch.cuda.get_device_name())


# In[231]:


# ============================================================
# Hyperparameters
# ============================================================

batch_size = 32
context_size = 128

max_iters = 3000
eval_interval = 300

learning_rate = 3e-4
dropout = 0.2

eval_iters = 100

n_embd = 192
n_heads = 6
n_layers = 4


# In[232]:


# ============================================================
# Data
# ============================================================

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

print(f"Vocab Size: {vocab_size}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9 * len(data))

train = data[:n]
val = data[n:]

print(train.shape)
print(val.shape)


# In[233]:


# ============================================================
# Batch Loader
# ============================================================

def get_batch(split):

    dataset = train if split == "train" else val

    ix = torch.randint(len(dataset) - context_size, (batch_size,))

    x = torch.stack(
        [dataset[i:i + context_size] for i in ix]
    )

    y = torch.stack(
        [dataset[i + 1:i + context_size + 1] for i in ix]
    )

    x = x.to(device)
    y = y.to(device)

    return x, y


# In[234]:


# ============================================================
# Attention Head
# ============================================================

class Head(nn.Module):
    """Single self-attention head"""

    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(context_size, context_size))
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        weight = (
            q @ k.transpose(-2, -1)
        ) * (k.shape[-1] ** -0.5)

        weight = weight.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)

        v = self.value(x)

        out = weight @ v

        return out

# ============================================================
# Multi Head Attention
# ============================================================

class MultiHeadAttention(nn.Module):

    def __init__(self, n_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList(
            [Head(head_size) for _ in range(n_heads)]
        )

        self.proj = nn.Linear(
            head_size * n_heads,
            n_embd
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        out = torch.cat(
            [h(x) for h in self.heads],
            dim=-1
        )

        out = self.proj(out)
        out = self.dropout(out)

        return out

# ============================================================
# Feed Forward
# ============================================================

class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# ============================================================
# Transformer Block
# ============================================================

class Block(nn.Module):

    def __init__(self, n_embd, n_heads):
        super().__init__()

        head_size = n_embd // n_heads

        self.sa = MultiHeadAttention(
            n_heads,
            head_size
        )

        self.ffwd = FeedForward(n_embd)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):

        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        return x

# ============================================================
# GPT Model
# ============================================================

class BigramLM(nn.Module):

    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(
            vocab_size,
            n_embd
        )

        self.position_embedding_table = nn.Embedding(
            context_size,
            n_embd
        )

        self.blocks = nn.Sequential(
            *[
                Block(n_embd, n_heads)
                for _ in range(n_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(n_embd)

        self.lm_head = nn.Linear(
            n_embd,
            vocab_size
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):

            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):

            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    def forward(self, idx, targets=None):

        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)

        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device)
        )

        x = tok_emb + pos_emb

        x = self.blocks(x)

        x = self.ln_f(x)

        logits = self.lm_head(x)

        if targets is None:
            loss = None

        else:

            B, T, C = logits.shape

            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(
                logits,
                targets
            )

        return logits, loss

    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):

            idx_cond = idx[:, -context_size:]

            logits, _ = self(idx_cond)

            logits = logits[:, -1, :]

            probs = F.softmax(
                logits,
                dim=-1
            )

            idx_next = torch.multinomial(
                probs,
                num_samples=1
            )

            idx = torch.cat(
                (idx, idx_next),
                dim=1
            )

        return idx

# ============================================================
# Loss Estimation
# ============================================================

@torch.no_grad()
def estimate_loss(model: nn.Module):

    out = {}

    model.eval()

    for split in ["train", "val"]:

        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):

            X, Y = get_batch(split)

            _, loss = model(X, Y)

            losses[k] = loss.item()

        out[split] = losses.mean()

    model.train()

    return out


# In[235]:


# ============================================================
# Model
# ============================================================

model = BigramLM().to(device)

print(
    sum(p.numel() for p in model.parameters()) / 1e6,
    "M parameters"
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)


# In[236]:


# ============================================================
# Training
# ============================================================

for iter in range(max_iters):

    if iter % eval_interval == 0 or iter == max_iters - 1:

        losses = estimate_loss(model)

        print(
            f"step {iter}: "
            f"train loss {losses['train']:.4f}, "
            f"val loss {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")

    _, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)

    loss.backward()

    optimizer.step()


# In[237]:


# ============================================================
# Generation
# ============================================================

context = torch.zeros(
    (1, 1),
    dtype=torch.long,
    device=device
)

generated = model.generate(
    context,
    max_new_tokens=500
)[0].tolist()

print(decode(generated))

