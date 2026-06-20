#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[2]:


import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name())

torch.manual_seed(1234)
device = "cuda" if torch.cuda.is_available() else "cpu"
device


# In[ ]:





# In[3]:


# Hyperparameters
train_split = 9
val_split = 1
eval_iters = 200

train_split = train_split/(train_split + val_split)
val_split = val_split/(train_split + val_split)


# In[4]:


with open('input.txt', 'r', encoding='utf-8') as f:
  text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

print(f'{" ".join(chars)}')
print(vocab_size)


# In[5]:


stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

encode('aads')
decode(encode('aads'))


# In[6]:


data = torch.tensor(encode(text), dtype=torch.long)

n = int(len(data) * train_split)
train = data[:n]
val = data[n:]

data.shape
train.shape
val.shape

train.shape[0] + val.shape[0]


# In[7]:


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


# In[8]:


import torch.nn as nn
from torch.nn import functional as F

class BiagramLM(nn.Module):
  def __init__(self, vocab_size, embed_dim=32, context_size=8):
    super().__init__()
    self.vocab_size = vocab_size
    self.embed_dim = embed_dim
    self.embeddings = nn.Embedding(vocab_size, vocab_size)
    # self.lm_head = nn.Linear(context_size, embed_dim)o

  def forward(self, idx, targets=None):
    logits = self.embeddings(idx)

    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits_transformed = logits.view(B*T, C)
      targets_transformed = targets.view(B*T)

      loss = F.cross_entropy(logits_transformed, targets_transformed)

    return logits, loss

  def generate(self, idx, max_new_tokens=50):
    for _ in range(max_new_tokens):
      logits, _ = self(idx)
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
      X, Y = get_batch(data=data)
      logits, loss = model(X, Y)
      losses[k] = loss.item()
    out[split] = losses.mean()

  model.train()
  return out


model = BiagramLM(vocab_size=vocab_size)

model.to(device)


# In[9]:


learning_rate = 1e-3
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

max_iters = 10000
eval_interval = 100

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


# In[48]:


context = torch.zeros((1, 1), dtype=torch.long, device=device)
raw_response = model.generate(context, max_new_tokens=50)[0].tolist()
response = decode(raw_response)

print(f"Response: {response}")

