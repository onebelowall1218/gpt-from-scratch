#!/usr/bin/env python
# coding: utf-8

# In[100]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[101]:


import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name())

torch.manual_seed(1234)
device = "cuda" if torch.cuda.is_available() else "cpu"
device


# In[ ]:





# In[102]:


# Hyperparameters
train_split = 9
val_split = 1
eval_iters = 200

train_split = train_split/(train_split + val_split)
val_split = val_split/(train_split + val_split)


# In[103]:


with open('input.txt', 'r', encoding='utf-8') as f:
  text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

print(f'{" ".join(chars)}')
print(vocab_size)


# In[104]:


stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

encode('aads')
decode(encode('aads'))


# In[105]:


data = torch.tensor(encode(text), dtype=torch.long)

n = int(len(data) * train_split)
train = data[:n]
val = data[n:]

data.shape
train.shape
val.shape

train.shape[0] + val.shape[0]


# In[106]:


def get_batch(data, batch_size=4, context_size=8):
  ix = torch.randint(len(data)-context_size, (batch_size,))
  x = torch.stack([data[i:i+context_size] for i in ix])
  y = torch.stack([data[i+1:i+context_size+1] for i in ix])
  x, y = x.to(device), y.to(device)
  return x, y

x, y = get_batch(data)
x.shape
y.shape
x
y


# In[107]:


import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
  """ A single head of the self attention model"""
  def __init__(self, context_size=8, n_embd=32, head_size=64):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(context_size, context_size)))

  def forward(self, x):
    B, T, C = x.shape
    k = self.key(x)
    q = self.query(x)
    # compute attention sccores 'affinities'
    weight = q @ k.transpose(-2, -1) * (k.size(-1) ** -0.5)
    weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    weight = F.softmax(weight, dim=-1)
    # perform the weighted aggregation of values
    v = self.value(x)

    out  = weight @ v

    return out

class MultiHeadAttention(nn.Module):
  """Multiple blocks of the self-attention blocks"""
  def __init__(self, n_heads, context_size=8, n_embd=32, head_size=64):
    super().__init__()
    self.heads = nn.ModuleList([Head(context_size, n_embd, head_size) for _ in range(n_heads)])

  def forward(self, x):
    return torch.cat([h(x) for h in self.heads], dim=-1)

class FeedForward(nn.Module):
  """A single layer of feed forward network"""
  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embd, n_embd),
      nn.ReLU(),
    )

  def forward(self, x):
    return self.net(x)


class BigramLM(nn.Module):
  def __init__(self, vocab_size, embed_dim=32, context_size=8):
    super().__init__()
    self.vocab_size = vocab_size
    self.embed_dim = embed_dim
    self.context_size = context_size

    self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)
    self.position_embedding_table = nn.Embedding(context_size, embed_dim)
    self.self_attention_heads = MultiHeadAttention(4, context_size, embed_dim, embed_dim//4)
    self.feed_forward_network = FeedForward(embed_dim)
    self.lm_head = nn.Linear(embed_dim, vocab_size)

  def forward(self, idx, targets=None):
    B, T = idx.shape

    token_embedding = self.token_embedding_table(idx) # (B, T, C)
    positional_embedding = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
    x = token_embedding + positional_embedding
    x = self.self_attention_heads(x) # (B, T, C)
    logits = self.lm_head(x) # (B, T, vocab_size)

    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits_transformed = logits.reshape(B*T, C)
      targets_transformed = targets.reshape(B*T)

      loss = F.cross_entropy(logits_transformed, targets_transformed)

    return logits, loss

  def generate(self, idx, max_new_tokens=50):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
      idx_continued = idx[:, -self.context_size:]
      logits, loss = self(idx_continued)
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)

    return idx

@torch.no_grad()
def estimate_loss(model):
  out = {}
  model.eval()

  for split in ['train', 'val']:
    dataset = train if split == 'train' else val
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = get_batch(data=dataset)
      logits, loss = model(X, Y)
      losses[k] = loss.item()
    out[split] = losses.mean()

  model.train()
  return out


model = BigramLM(vocab_size=vocab_size)

model.to(device)


# In[108]:


learning_rate = 1e-3
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

max_iters = 5000
eval_interval = 500

for iter in range(max_iters):
  if iter % eval_interval == 0:
    losses = estimate_loss(model)
    if isinstance(losses, dict):
      print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']: .4f}")
    else:
      print(f"No validation set, so no validation loss.")

  xb, yb = get_batch(data=data)
  logits, loss = model(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()


# In[109]:


context = torch.zeros((1, 1), dtype=torch.long, device=device)
raw_response = model.generate(context, max_new_tokens=200)[0].tolist()
response = decode(raw_response)

print(f"Response: {response}")


# In[110]:


# self-attention

B, T, C = 4, 8, 32

x = torch.randn(B, T, C)

head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
k = key(x)
q = query(x)

weight = q @ k.transpose(-2, -1) # (B, T, 16) @ (B, 16, T) = (B, T, T) 

tril = torch.tril(torch.ones(T, T))
# weight = torch.zeros((T, T))
weight = weight.masked_fill(tril == 0, float("-inf"))
weight = F.softmax(weight, dim=-1)

out = weight @ x

out.shape
weight[0]


# In[111]:


k = torch.randn(B, T, head_size)
q = torch.randn(B, T, head_size)

weight = q @ k.transpose(-2, -1) * (head_size ** -0.5) # (B, T, 16) @ (B, 16, T) = (T, T)

k.var()
q.var()
weight.var()

