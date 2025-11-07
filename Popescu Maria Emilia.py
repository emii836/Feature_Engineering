
#problema 1
def generate_square_subsequent_mask(sz):
    """Creeaza o mască triangulara inferioara pentru a ascunde tokenii viitori."""
    mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
    return mask


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(0.1)

        common_args = {
            "d_model": d_model,
            "nhead": nhead,
            "dim_feedforward": 128,
            "batch_first": True
        }

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(**common_args),
            num_layers=num_layers
        )

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(**common_args),
            num_layers=num_layers
        )

        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, src_ids, tgt_ids):
        # Embedding + pozițional encoding
        src = self.dropout(self.pos_enc(self.tok_emb(src_ids) * math.sqrt(self.d_model)))
        tgt = self.dropout(self.pos_enc(self.tok_emb(tgt_ids) * math.sqrt(self.d_model)))

        # Măști pentru padding
        src_pad_mask = (src_ids == stoi["<pad>"])
        tgt_pad_mask = (tgt_ids == stoi["<pad>"])

       
        tgt_seq_len = tgt_ids.size(1)
        tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(tgt_ids.device)

        # Encoder + Decoder
        memory = self.encoder(src, src_key_padding_mask=src_pad_mask)
        out = self.decoder(
            tgt, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask
        )

        return self.out(out)


#  model + pierdere + optimizer
model = SimpleTransformer(len(VOCAB)).to(device)
BATCH, EPOCHS = 32, 10
criterion = nn.CrossEntropyLoss(ignore_index=stoi["<pad>"])
optimizer = optim.Adam(model.parameters(), lr=1e-3)

#problema 2
def train_epoch():
    model.train()
    total_loss = 0.0
    for i in range(0, train_X.size(0), BATCH):
        src, tgt = train_X[i:i + BATCH].to(device), train_Y[i:i + BATCH].to(device)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]

        out = model(src, tgt_in)  # (batch, seq_len, vocab_size)
        out_flat = out.reshape(-1, out.size(-1))  # (batch*seq_len, vocab_size)
        tgt_flat = tgt_out.reshape(-1)            # (batch*seq_len)

        loss = criterion(out_flat, tgt_flat)     

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * src.size(0)
    return total_loss / train_X.size(0)



