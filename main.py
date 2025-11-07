import random, math, torch, torch.nn as nn, torch.optim as optim
def generate_square_subsequent_mask(sz):
    """Creeaza o masca triangulara inferioara pentru a ascunde viitorul."""
    mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
    return mask

SEED = 2025
random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DIGITS = {"ZERO": 0, "ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4, "FIVE": 5, "SIX": 6, "SEVEN": 7, "EIGHT": 8, "NINE": 9}
OPS = ["+", "-"]
VOCAB = ["<pad>", "<sos>", "<eos>"] + list(DIGITS.keys()) + OPS + [str(i) for i in range(20)]
stoi = {s: i for i, s in enumerate(VOCAB)}

def gen_example(max_len=3):
    n_terms = random.randint(1, max_len)
    tokens, values = [], []
    for i in range(n_terms):
        w = random.choice(list(DIGITS.keys()))
        tokens.append(w)
        values.append(DIGITS[w])
        if i < n_terms - 1: tokens.append(random.choice(OPS))
    acc = values[0]
    for j in range(1, len(tokens), 2):
        acc += DIGITS[tokens[j + 1]] if tokens[j] == "+" else -DIGITS[tokens[j + 1]]
    return tokens, list(str(max(0, min(19, acc))))

def encode(tokens, max_len):
    ids = [stoi["<sos>"]] + [stoi.get(t, stoi["<pad>"]) for t in tokens] + [stoi["<eos>"]]
    if len(ids) < max_len:
        ids += [stoi["<pad>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
        ids[-1] = stoi["<eos>"]
    return ids


train_data = [gen_example(max_len=3) for _ in range(2000)]
val_data = [gen_example(max_len=3) for _ in range(400)]
train_X = torch.tensor([encode(s, 10) for s, t in train_data], dtype=torch.long)
train_Y = torch.tensor([encode(t, 5) for s, t in train_data], dtype=torch.long)
val_X = torch.tensor([encode(s, 10) for s, t in val_data], dtype=torch.long)
val_Y = torch.tensor([encode(t, 5) for s, t in val_data], dtype=torch.long)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(0.1)
        common_args = {"d_model": d_model, "nhead": nhead, "dim_feedforward": 128, "batch_first": True}
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(**common_args), num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(**common_args), num_layers=num_layers)
        self.out = nn.Linear(d_model, vocab_size)
    def forward(self, src_ids, tgt_ids):
    src = self.dropout(self.pos_enc(self.tok_emb(src_ids) * math.sqrt(self.d_model)))
    tgt = self.dropout(self.pos_enc(self.tok_emb(tgt_ids) * math.sqrt(self.d_model)))
    src_pad_mask = (src_ids == stoi["<pad>"])
    tgt_pad_mask = (tgt_ids == stoi["<pad>"])
    
 
    tgt_seq_len = tgt_ids.size(1)
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(tgt_ids.device)
    
    memory = self.encoder(src, src_key_padding_mask=src_pad_mask)
    out = self.decoder(
        tgt, memory,
        tgt_mask=tgt_mask,
        tgt_key_padding_mask=tgt_pad_mask,
        memory_key_padding_mask=src_pad_mask
    )
    return self.out(out)


model = SimpleTransformer(len(VOCAB)).to(device)
BATCH, EPOCHS = 32, 10
criterion = nn.CrossEntropyLoss(ignore_index=stoi["<pad>"])
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train_epoch():
    model.train()
    total_loss = 0.0
    for i in range(0, train_X.size(0), BATCH):
        src, tgt = train_X[i:i + BATCH].to(device), train_Y[i:i + BATCH].to(device)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        out = model(src, tgt_in)
        out_flat, tgt_flat = out.reshape(-1, out.size(-1)), tgt_out.reshape(-1)
         loss = criterion(out_flat, tgt_flat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * src.size(0)
    return total_loss / train_X.size(0)

@torch.no_grad()

def evaluate():
    model.eval()
    correct, total = 0, 0
    for i in range(0, val_X.size(0), BATCH):
        src, tgt = val_X[i:i + BATCH].to(device), val_Y[i:i + BATCH].to(device)
        bs = src.size(0)
        dec_in = torch.full((bs, 1), stoi["<sos>"], dtype=torch.long).to(device)
        for _ in range(tgt.size(1) - 1):
            out = model(src, dec_in)
            next_tok = out[:, -1, :].argmax(dim=-1, keepdim=True)
            dec_in = torch.cat([dec_in, next_tok], dim=1)
        mask = tgt != stoi["<pad>"]
        correct += ((dec_in[:, 1:tgt.size(1)] == tgt[:, 1:]) & mask[:, 1:]).sum().item()
        total += mask[:, 1:].sum().item()
    return 100.0 * correct / total


for ep in range(1, EPOCHS + 1):
    print(f"Epoch {ep}/{EPOCHS}  Loss={train_epoch():.4f}  ValAcc={evaluate():.2f}%")

final_acc = evaluate()
print(f"Final acc: {final_acc:.2f}%")
if final_acc >= 95.0:
    print(" Modelul a atins pragul!")
else:
    print(" Modelul nu a atins pragul — Debug & Retry!")
