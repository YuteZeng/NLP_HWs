#!/usr/bin/python

"""
Implement a trigrm HMM here. 
You model should output the HMM similar to `train_hmm.py`.

Usage:  python train_trigram_hmm.py tags text > hmm-file

"""

# To modify the original bigram model to trigram
# Change the state of the 'prev' word into 'prevprev_prev' word combination

import sys, re
from collections import defaultdict

TAG_FILE = sys.argv[1]
TOKEN_FILE = sys.argv[2]

vocab = {}
OOV_WORD = "OOV"
INIT_STATE = "init"
FINAL_STATE = "final"

emissions = {}
transitions = {}
transitionsTotal = defaultdict(int)
emissionsTotal = defaultdict(int)

with open(TAG_FILE) as tagFile, open(TOKEN_FILE) as tokenFile:
    for tagString, tokenString in zip(tagFile, tokenFile):

        tags = re.split("\s+", tagString.rstrip())
        tokens = re.split("\s+", tokenString.rstrip())
        pairs = list(zip(tags, tokens))

        prevtag = INIT_STATE
        prevprevtag = INIT_STATE

        for (tag, token) in pairs:

            if token not in vocab:
                vocab[token] = 1
                token = OOV_WORD

            trigram_transition_key = (prevprevtag, prevtag)

            if tag not in emissions:
                emissions[tag] = defaultdict(int)
            if trigram_transition_key not in transitions:
                transitions[trigram_transition_key] = defaultdict(int)

            emissions[tag][token] += 1
            emissionsTotal[tag] += 1

            transitions[trigram_transition_key][tag] += 1
            transitionsTotal[trigram_transition_key] += 1

            prevprevtag = prevtag
            prevtag = tag

        trigram_transition_key = (prevprevtag, prevtag)
        if trigram_transition_key not in transitions:
            transitions[trigram_transition_key] = defaultdict(int)
        
        transitions[trigram_transition_key][FINAL_STATE] += 1
        transitionsTotal[trigram_transition_key] += 1

# Outputting the transition and emission probabilities
for (prevprevtag, prevtag) in transitions:
    for tag in transitions[(prevprevtag, prevtag)]:
        print(f"trans {prevprevtag} {prevtag} {tag} {float(transitions[(prevprevtag, prevtag)][tag]) / transitionsTotal[(prevprevtag, prevtag)]}")

for tag in emissions:
    for token in emissions[tag]:
        print(f"emit {tag} {token} {float(emissions[tag][token]) / emissionsTotal[tag]}")
