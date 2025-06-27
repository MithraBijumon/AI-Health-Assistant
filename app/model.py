import os, pickle, torch
from typing import List

base_dir = os.path.dirname(os.path.abspath(__file__))

# ── load artefacts ────────────────────────────────────────────────
with open(os.path.join(base_dir, "symptom_vocab.pkl"), "rb") as f:
    symptom_vocab = pickle.load(f)

with open(os.path.join(base_dir, "disease_vocab.pkl"), "rb") as f:
    disease_vocab = pickle.load(f)

with open(os.path.join(base_dir, "config.pkl"), "rb") as f:
    cfg = pickle.load(f)

MAX_LENGTH = cfg["MAX_LENGTH"]
PAD_token  = cfg["PAD_token"]
EOS_token  = cfg["EOS_token"]
SOS_token  = cfg["SOS_token"]

# ── hyper-parameters (same as training) ────────────────────────────
hidden_size = 128          # << set exactly to the value used in train_model.py
device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── model classes (same code you used in training) ──
import torch.nn as nn
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super().__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.dropout   = nn.Dropout(dropout_p)
        self.gru       = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        out, h = self.gru(emb)
        return out, h


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, q, k):                       # q: [B,1,H], k:[B,T,H]
        scores  = self.Va(torch.tanh(self.Wa(q) + self.Ua(k)))   # [B,T,1]
        weights = F.softmax(scores.squeeze(2), dim=-1).unsqueeze(1)
        ctx     = torch.bmm(weights, k)            # [B,1,H]
        return ctx, weights


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super().__init__()
        self.embedding  = nn.Embedding(output_size, hidden_size)
        self.attention       = BahdanauAttention(hidden_size)
        self.gru        = nn.GRU(hidden_size*2, hidden_size, batch_first=True)
        self.out        = nn.Linear(hidden_size, output_size)
        self.dropout    = nn.Dropout(dropout_p)

    def forward(self, enc_out, enc_hidden, target_tensor=None):
        B = enc_out.size(0)
        dec_input  = torch.full((B,1), SOS_token, dtype=torch.long, device=device)
        dec_hidden = enc_hidden
        outputs    = []
        max_len    = target_tensor.size(1) if target_tensor is not None else MAX_LENGTH

        for t in range(max_len):
            emb = self.dropout(self.embedding(dec_input))        # [B,1,H]
            ctx, _ = self.attention(dec_hidden.permute(1,0,2), enc_out)
            gru_in = torch.cat([emb, ctx], dim=2)
            dec_out, dec_hidden = self.gru(gru_in, dec_hidden)   # [B,1,H]
            logits = self.out(dec_out)                           # [B,1,V]
            outputs.append(logits)

            if target_tensor is not None:
                dec_input = target_tensor[:, t].unsqueeze(1)
            else:
                dec_input = logits.argmax(-1)

        return torch.cat(outputs, 1), dec_hidden

# ── rebuild model ─────────────────────────────────────────────────
encoder = EncoderRNN(len(symptom_vocab["div2index"]), hidden_size).eval()
decoder = AttnDecoderRNN(hidden_size, len(disease_vocab["div2index"])).eval()

encoder.load_state_dict(torch.load(os.path.join(base_dir, "encoder.pt")))
decoder.load_state_dict(torch.load(os.path.join(base_dir, "decoder.pt")))

# ── helpers ───────────────────────────────────────────────────────
import re
from nltk.corpus import stopwords

import nltk
import contractions
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
lemmatizer = WordNetLemmatizer()

class LemmatizationWithPOSTagger(object):
    def __init__(self):
        pass

    def get_wordnet_pos(self,treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def pos_tag(self,tokens):
        out = []
        tags = pos_tag(tokens)
        for token, POS in tags:
            out.append(lemmatizer.lemmatize(token, pos=self.get_wordnet_pos(POS)))
        return out

stop_words = set(stopwords.words('english'))
lemmatization_using_pos_tagger = LemmatizationWithPOSTagger()

def normalise(text: str) -> List[str]:
    text = contractions.fix(text)
    text = text.lower().strip()
    text = re.sub(r"([.!?])", r" \1", text)
    text = re.sub(r"[^a-zA-Z!?]+", r" ", text)
    text = text.split()
    text = lemmatization_using_pos_tagger.pos_tag(text)
    filtered_sentence = [w for w in text if not w in stop_words]
    print(filtered_sentence)
    return filtered_sentence    

def encode(symptom_sentence: str):
    idxs = [
        symptom_vocab["div2index"].get(tok, 1)  # 1 = UNK_token
        for tok in normalise(symptom_sentence)
    ][:MAX_LENGTH-1] + [EOS_token]              # clamp + add EOS
    idxs += [PAD_token] * (MAX_LENGTH - len(idxs))
    return torch.tensor(idxs, dtype=torch.long).unsqueeze(0)  # [1, T]

@torch.no_grad()
def diagnose(sentence: str) -> str:
    x = encode(sentence)                  # [1, T]
    enc_out, enc_hidden = encoder(x)

    dec_input  = torch.full((1, 1), SOS_token, dtype=torch.long, device=device)
    dec_hidden = enc_hidden
    words      = []
    seen       = set()  # To track generated tokens

    for _ in range(MAX_LENGTH):
        emb = decoder.embedding(dec_input)
        ctx, _ = decoder.attention(dec_hidden.permute(1,0,2), enc_out)
        gru_in = torch.cat([emb, ctx], dim=2)
        dec_out, dec_hidden = decoder.gru(gru_in, dec_hidden)
        logits = decoder.out(dec_out)                 # [1,1,V]
        topi   = logits.argmax(-1)                    # [1,1]

        token  = topi.item()
        if token == EOS_token:
            break
        word = disease_vocab["index2div"][token]
        
        if word not in seen:
            words.append(word)
            seen.add(word)

        dec_input = topi.detach()  # feed back

    return " ".join(words) if words else "<UNK>"


# ── quick test ────────────────────────────────────────────────────
if __name__ == "__main__":
    print(diagnose("I have itching and vomiting with yellowish skin"))

