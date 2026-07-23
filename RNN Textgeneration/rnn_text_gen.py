"""
RNN Text Generation (Character-Level LSTM)
============================================
Trains a character-level RNN (LSTM) to generate text one character
at a time, learning patterns directly from a training text.

Dataset: a public-domain text sample is embedded directly in this
file (see TEXT_DATA below) so there's NOTHING to download — this
avoids the slow-download issue from CIFAR-10. You can freely swap
in your own text (a book, lyrics you own, log files, etc.) by
replacing TEXT_DATA or loading a .txt file — see the bottom of this
section for how.

Requirements:
    pip install torch --break-system-packages

Run:
    python rnn_text_gen.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random

# ---------------------------------------------------------
# 1. Dataset (embedded — no download needed)
# ---------------------------------------------------------
# A public-domain excerpt (Lewis Carroll, "The Jabberwocky", 1871 —
# old enough to be public domain everywhere). Swap this out for any
# text you like; more text = better results (10,000+ characters is
# a reasonable minimum for anything beyond a toy demo).
TEXT_DATA = """
'Twas brillig, and the slithy toves
Did gyre and gimble in the wabe:
All mimsy were the borogoves,
And the mome raths outgrabe.

"Beware the Jabberwock, my son!
The jaws that bite, the claws that catch!
Beware the Jubjub bird, and shun
The frumious Bandersnatch!"

He took his vorpal sword in hand;
Long time the manxome foe he sought—
So rested he by the Tumtum tree,
And stood awhile in thought.

And, as in uffish thought he stood,
The Jabberwock, with eyes of flame,
Came whiffling through the tulgey wood,
And burbled as it came!

One, two! One, two! And through and through
The vorpal blade went snicker-snack!
He left it dead, and with its head
He went galumphing back.

"And hast thou slain the Jabberwock?
Come to my arms, my beamish boy!
O frabjous day! Callooh! Callay!"
He chortled in his joy.

'Twas brillig, and the slithy toves
Did gyre and gimble in the wabe:
All mimsy were the borogoves,
And the mome raths outgrabe.
""" * 20  # repeated to give the tiny demo dataset enough volume to learn from

# ---- To use your OWN text file instead, replace the block above with: ----
# with open("your_text_file.txt", "r", encoding="utf-8") as f:
#     TEXT_DATA = f.read()
# ---------------------------------------------------------------------------

print(f"Dataset length: {len(TEXT_DATA):,} characters")

# ---------------------------------------------------------
# 2. Build vocabulary (character-level)
# ---------------------------------------------------------
chars = sorted(list(set(TEXT_DATA)))
VOCAB_SIZE = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

print(f"Vocabulary size: {VOCAB_SIZE} unique characters")
print(f"Characters: {''.join(chars)!r}")


def encode(text):
    return [char_to_idx[ch] for ch in text]


def decode(indices):
    return "".join(idx_to_char[i] for i in indices)


data = torch.tensor(encode(TEXT_DATA), dtype=torch.long)

# ---------------------------------------------------------
# 3. Config
# ---------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LENGTH = 100        # characters per training sequence
BATCH_SIZE = 64
EMBED_DIM = 64
HIDDEN_DIM = 256
NUM_LAYERS = 2
NUM_EPOCHS = 30
LEARNING_RATE = 3e-3
MODEL_SAVE_PATH = "./rnn_text_gen.pth"

print(f"Using device: {DEVICE}")


# ---------------------------------------------------------
# 4. Batch generation
# ---------------------------------------------------------
def get_batch(data, seq_length=SEQ_LENGTH, batch_size=BATCH_SIZE):
    """Randomly sample `batch_size` sequences of length `seq_length`.
    Input is chars[i:i+seq_length], target is the same span shifted
    by one character (predict-the-next-character)."""
    max_start = len(data) - seq_length - 1
    starts = torch.randint(0, max_start, (batch_size,))

    x = torch.stack([data[s : s + seq_length] for s in starts])
    y = torch.stack([data[s + 1 : s + seq_length + 1] for s in starts])

    return x.to(DEVICE), y.to(DEVICE)


# ---------------------------------------------------------
# 5. Model: character-level LSTM
# ---------------------------------------------------------
class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        # x: (batch, seq_len) of token indices
        embedded = self.embedding(x)                    # (batch, seq_len, embed_dim)
        output, hidden = self.lstm(embedded, hidden)     # (batch, seq_len, hidden_dim)
        logits = self.fc(output)                         # (batch, seq_len, vocab_size)
        return logits, hidden

    def init_hidden(self, batch_size):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(DEVICE)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(DEVICE)
        return (h0, c0)


model = CharRNN(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ---------------------------------------------------------
# 6. Training loop
# ---------------------------------------------------------
STEPS_PER_EPOCH = max(1, len(TEXT_DATA) // (SEQ_LENGTH * BATCH_SIZE))


def train():
    model.train()
    losses = []

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_loss = 0.0

        for step in range(STEPS_PER_EPOCH):
            x, y = get_batch(data)
            hidden = model.init_hidden(x.size(0))
            # Detach hidden state so gradients don't flow across batches
            hidden = tuple(h.detach() for h in hidden)

            optimizer.zero_grad()
            logits, hidden = model(x, hidden)

            loss = criterion(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / STEPS_PER_EPOCH
        losses.append(avg_loss)
        print(f"Epoch {epoch}/{NUM_EPOCHS} | Loss: {avg_loss:.4f}")

        if epoch % 5 == 0 or epoch == NUM_EPOCHS:
            sample = generate(seed_text="The ", length=150, temperature=0.8)
            print(f"  Sample: {sample!r}\n")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    return losses


# ---------------------------------------------------------
# 7. Text generation (sampling with temperature)
# ---------------------------------------------------------
@torch.no_grad()
def generate(seed_text="The ", length=200, temperature=0.8):
    """
    temperature < 1.0 -> more conservative / repetitive text
    temperature > 1.0 -> more random / creative (and more mistakes)
    """
    model.eval()

    # Fall back to a known character if the seed has something unseen
    seed_text = "".join(ch if ch in char_to_idx else " " for ch in seed_text)

    input_seq = torch.tensor([encode(seed_text)], dtype=torch.long).to(DEVICE)
    hidden = model.init_hidden(1)

    generated = seed_text

    # Warm up hidden state on the seed
    _, hidden = model(input_seq, hidden)
    last_char = input_seq[:, -1:]

    for _ in range(length):
        logits, hidden = model(last_char, hidden)
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1)

        generated += idx_to_char[next_idx.item()]
        last_char = next_idx

    model.train()
    return generated


# ---------------------------------------------------------
# 8. Load a saved model later (for generation without retraining)
# ---------------------------------------------------------
def load_and_generate(seed_text="The ", length=200, temperature=0.8,
                       model_path=MODEL_SAVE_PATH):
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    return generate(seed_text, length, temperature)


if __name__ == "__main__":
    train()

    print("\n=== Final generation samples at different temperatures ===")
    for temp in [0.5, 0.8, 1.2]:
        text = generate(seed_text="'Twas ", length=200, temperature=temp)
        print(f"\n--- Temperature {temp} ---\n{text}")
