#!/usr/bin/env python
# coding: utf-8

# In[97]:


# 1. Requirements
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[147]:


# 2. Load dataset
# one
df1 = pd.read_csv(r"C:\Users\mithr\Downloads\dataset.csv")

sym_cols = ["Symptom_1", "Symptom_2", "Symptom_3", "Symptom_4"]

# Join non-empty cells with commas:
df1["symptoms"] = (
    df1[sym_cols]
      .fillna("")                       # replace NaN with empty string
      .astype(str)
      .apply(lambda row: ", ".join([s.strip() for s in row if s.strip()]), axis=1)
)

symptoms = df1["symptoms"].tolist()

diseases = df1["Disease"].fillna("").astype(str).tolist()

#two
df2 = pd.read_csv(r"C:\Mithra\WNCC\health-assistant\app\Diseases_Symptoms.csv")
symptoms.extend(df2["Symptoms"].fillna("").astype(str))
diseases.extend(df2["Name"].fillna("").astype(str))

df3 = pd.read_csv(r"C:\Mithra\WNCC\health-assistant\app\Disease_symptom_and_patient_profile_dataset.csv")
for idx, row in df3.iterrows():
    if row["Outcome Variable"] == "Positive":
        diseases.extend(df3["Disease"].fillna("").astype(str))
        s = []
        for i in "Fever,Cough,Fatigue,Difficulty Breathing,Age,Gender,Blood Pressure,Cholesterol Level".split(","):
            s.append(i)
        symptoms.extend(s)


# In[148]:
import nltk
from nltk.corpus import stopwords
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

# Turn a Unicode string to plain ASCII
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    s = s.strip()
    s = s.split()
    s = lemmatization_using_pos_tagger.pos_tag(s)
    filtered_sentence = [w for w in s if not w in stop_words]
    return " ".join(filtered_sentence)


# In[149]:


# 3. Make Vocabulary
SOS_token = 0
UNK_token = 1
EOS_token = 2
PAD_token = 3

class Record:
    def __init__(self, name):
        self.name = name
        self.div2index = {"SOS":0, "UNK":1, "EOS": 2, "PAD":3}
        self.div2count = {}
        self.index2div = {0: "SOS", 1: "UNK", 2: "EOS", 3: "PAD"}
        self.n_divs = 4  

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addDiv(word)

    def addDiv(self, word):
        if word not in self.div2index:
            self.div2index[word] = self.n_divs
            self.div2count[word] = 1
            self.index2div[self.n_divs] = word
            self.n_divs += 1
        else:
            self.div2count[word] += 1


# In[150]:


# 4. Read Data
def readRecords(div1, div2):
    print("Reading records...")

    # Split every line into pairs
    pairs = [(normalizeString(s), normalizeString(d))
         for s, d in zip(symptoms, diseases)]


    # Make Record instances
    input_symptom = Record(div1)
    output_disease = Record(div2)

    return input_symptom, output_disease, pairs


# In[151]:


# 5. Filter Data
MAX_LENGTH = 25


def filterPair(p):
    if not all(isinstance(x, str) and x.strip() != "" for x in p):
        #print("Rejected (non-string or empty):", p)
        return False
    if any(len(x.split()) >= MAX_LENGTH for x in p):
        #print("Rejected (too long):", p)
        return False
    return True


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


# In[156]:


# 6. Preparing Data
def prepareData(div1, div2):
    input_symptom, output_disease, pairs = readRecords(div1, div2)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_symptom.addSentence(pair[0])
        output_disease.addSentence(pair[1])
    print("Counted words:")
    print(input_symptom.name, input_symptom.n_divs)
    print(output_disease.name, output_disease.n_divs)
    return input_symptom, output_disease, pairs

input_symptom, output_disease, pairs = prepareData('symptoms', 'diseases')
print(random.choice(pairs))


# In[64]:


# 7. Encoder Layer
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden


# In[136]:


# 8. Decoder Layer
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None, use_teacher_forcing=True):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        seq_len = target_tensor.size(1) if target_tensor is not None else MAX_LENGTH
        for i in range(seq_len):
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if use_teacher_forcing and target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden


# In[80]:


# 9. Attention Decoder
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None, use_teacher_forcing=True):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        seq_len = target_tensor.size(1) if target_tensor is not None else MAX_LENGTH
        for i in range(seq_len):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if use_teacher_forcing and target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights


# In[113]:


# 10. Preparing Training Data
def indexesFromSentence(div, sentence):
    return [div.div2index.get(word, UNK_token) for word in sentence.split()] + [EOS_token]

def tensorFromSentence(div, sentence):
    indexes = indexesFromSentence(div, sentence)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_symptom, pair[0])
    target1_tensor = tensorFromSentence(output_disease, pair[1])
    return (input_tensor, target1_tensor)

def get_dataloader(batch_size):
    input_symptom, output_disease, pairs = prepareData('symptoms', 'diseases')

    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_symptom, inp) + [EOS_token]
        tgt_ids = indexesFromSentence(output_disease, tgt) + [EOS_token]
        inp_ids = inp_ids[:MAX_LENGTH]
        inp_ids += [PAD_token]*(MAX_LENGTH-len(inp_ids))
        tgt_ids = tgt_ids[:MAX_LENGTH]
        tgt_ids += [PAD_token]*(MAX_LENGTH-len(tgt_ids))
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    train_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                               torch.LongTensor(target_ids).to(device)
                               )

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, drop_last=True, sampler=train_sampler, batch_size=batch_size)
    return input_symptom, output_disease, train_dataloader


# In[140]:


# 11. Training the Model
def mask_after_eos(target_tensor, pad_token=PAD_token, eos_token=EOS_token):
    # target_tensor: [B, T]
    B, T = target_tensor.size()
    mask = torch.ones_like(target_tensor, dtype=torch.bool)

    for b in range(B):
        found_eos = False
        for t in range(T):
            if found_eos or target_tensor[b, t] == pad_token:
                mask[b, t] = False
            if target_tensor[b, t] == eos_token:
                mask[b, t] = False
                found_eos = True
    return mask

def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion, epoch, n_epochs):

    total_loss = 0
    tf_ratio = max(0.1, 0.7 * (1 - epoch / n_epochs))
    for input_tensor, target_tensor in dataloader:
        use_tf = torch.rand(1).item() < tf_ratio

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, 
                                        target_tensor, use_teacher_forcing=use_tf)

        #print("decoder_outputs:", decoder_outputs.shape)
        #print("target1_tensor:", target1_tensor.shape)

        mask = mask_after_eos(target_tensor)
        logits = decoder_outputs.view(-1, decoder_outputs.size(-1))
        targets = target_tensor.view(-1)
        loss = criterion(logits[mask.view(-1)], targets[mask.view(-1)])

        loss.backward()

        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(dataloader))


# In[69]:


# 12. Print Time Elapsed and Estimated Time Remaining given the current time and progress %
import time
import math

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


# In[70]:


class DiseaseClassifier(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, vocab_size)
    def forward(self, enc_hidden):
        logits = self.fc(enc_hidden[-1])        # [B, vocab]
        return F.log_softmax(logits, dim=-1)


# In[71]:


# 13. Training
def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001,
               print_every=100, plot_every=100):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss(ignore_index=PAD_token)

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, encoder, decoder, 
                           encoder_optimizer, decoder_optimizer, criterion, epoch, n_epochs)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                        epoch, epoch / n_epochs * 100, print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)


# In[72]:


# 14. Plotting Results
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


# In[131]:


# 15. Evaluation
def evaluate(encoder, decoder, sentence, input_symptom, output_disease):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_symptom, sentence)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append('<PAD>')
                break
            decoded_words.append(output_disease.index2div[idx.item()])
    return decoded_words, decoder_attn


# In[167]:


from collections import OrderedDict
def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, _ = evaluate(encoder, decoder, pair[0], input_symptom, output_disease)
        output_sentence = ' '.join(OrderedDict.fromkeys(output_words))
        print('<', output_sentence)
        print('')


# In[157]:


hidden_size = 128
batch_size = 32

input_symptom, output_disease, train_dataloader = get_dataloader(batch_size)

encoder = EncoderRNN(input_symptom.n_divs, hidden_size).to(device)
decoder = AttnDecoderRNN(hidden_size, output_disease.n_divs).to(device)

train(train_dataloader, encoder, decoder, 20, print_every=5, plot_every=5)


# In[169]:


encoder.eval()
decoder.eval()
evaluateRandomly(encoder, decoder)


# In[ ]:
# ── SAVE artefacts so model.py can load without edits ─────────────
import os, pickle, torch

base_dir = os.path.dirname(os.path.abspath(__file__))

# Strip the heavy attributes – keep only what you need for tokenisation
symptom_vocab = {
    "div2index": input_symptom.div2index,
    "index2div": input_symptom.index2div
}
disease_vocab = {
    "div2index": output_disease.div2index,
    "index2div": output_disease.index2div
}

with open(os.path.join(base_dir, "symptom_vocab.pkl"), "wb") as f:
    pickle.dump(symptom_vocab, f)

with open(os.path.join(base_dir, "disease_vocab.pkl"), "wb") as f:
    pickle.dump(disease_vocab, f)

torch.save(encoder.state_dict(), os.path.join(base_dir, "encoder.pt"))
torch.save(decoder.state_dict(), os.path.join(base_dir, "decoder.pt"))


# Save a tiny json/txt with global constants
with open(os.path.join(base_dir, "config.pkl"), "wb") as f:
    pickle.dump(
        {"MAX_LENGTH": MAX_LENGTH,
         "PAD_token": PAD_token,
         "EOS_token": EOS_token,
         "SOS_token": SOS_token},
        f
    )

print("✔ saved symptom_vocab.pkl, disease_vocab.pkl, model.pt, config.pkl")



