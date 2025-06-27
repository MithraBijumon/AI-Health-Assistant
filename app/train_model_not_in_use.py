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

# 2. Load dataset
df = pd.read_csv("Diseases_Symptoms.csv")
symptoms = df["Symptoms"]
diseases = df["Name"]

# 3. Make Vocabulary
UNK_token = 0
PAD_token = 1

class Record:
    def __init__(self, name):
        self.name = name
        self.symptom2index = {}
        self.symptom2count = {}
        self.index2symptom = {0: "UNK", 1: "PAD"}
        self.n_symptoms = 2  # Count UNK and PAD

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addSymptom(word)

    def addSymptom(self, word):
        if word not in self.symptom2index:
            self.symptom2index[word] = self.n_symptoms
            self.symptom2count[word] = 1
            self.index2symptom[self.n_symptoms] = word
            self.n_symptoms += 1
        else:
            self.symptom2count[word] += 1

# 4. Preparing Data
