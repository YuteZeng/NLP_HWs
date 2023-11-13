#!/usr/bin/python

# David Bamman
# 2/14/14
#
# Python port of train_hmm.pl:

# Noah A. Smith
# 2/21/08
# Code for maximum likelihood estimation of a bigram HMM from
# column-formatted training data.

# Usage:  python train_hmm.py tags text > hmm-file

# The training data should consist of one line per sequence, with
# states or symbols separated by whitespace and no trailing whitespace.
# The initial and final states should not be mentioned; they are
# implied.
# The output format is the HMM file format as described in viterbi.pl.

import sys, re
import math
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


# function for changing the corpus size for training
# input: data from training data and tags
#        N is the training data size 
# output: store trained model in file named by size N
def train_HMM():
    vocab = {}
    OOV_WORD = "OOV"
    INIT_STATE = "init"
    FINAL_STATE = "final"

    emissions = {}
    transitions = {}
    transitionsTotal = defaultdict(int)
    emissionsTotal = defaultdict(int)

    sentence_count = 0 

    with open(TAG_FILE, encoding='utf-8') as tagFile, open(TOKEN_FILE, encoding='utf-8') as tokenFile:
        for tagString, tokenString in zip(tagFile, tokenFile):
            tags = re.split("\s+", tagString.rstrip())
            tokens = re.split("\s+", tokenString.rstrip())
            pairs = list(zip(tags, tokens))

            prevtag = INIT_STATE

            for (tag, token) in pairs:

                # this block is a little trick to help with out-of-vocabulary (OOV)
                # words.  the first time we see *any* word token, we pretend it
                # is an OOV.  this lets our model decide the rate at which new
                # words of each POS-type should be expected (e.g., high for nouns,
                # low for determiners).

                if token not in vocab:
                    vocab[token] = 1
                    token = OOV_WORD

                if tag not in emissions:
                    emissions[tag] = defaultdict(int)
                if prevtag not in transitions:
                    transitions[prevtag] = defaultdict(int)

                # increment the emission/transition observation
                emissions[tag][token] += 1
                emissionsTotal[tag] += 1

                transitions[prevtag][tag] += 1
                transitionsTotal[prevtag] += 1

                prevtag = tag

            # don't forget the stop probability for each sentence
            if prevtag not in transitions:
                transitions[prevtag] = defaultdict(int)

            transitions[prevtag][FINAL_STATE] += 1
            transitionsTotal[prevtag] += 1

    # model store
    model_path = 'btb.hmm'
    with open(model_path, 'a', encoding='utf-8') as file:
        for prevtag in transitions:
            for tag in transitions[prevtag]:
                file.write("trans %s %s %s\n" % (prevtag, tag, float(transitions[prevtag][tag]) / transitionsTotal[prevtag]))

        for tag in emissions:
            for token in emissions[tag]:
                file.write("emit %s %s %s\n" % (tag, token, float(emissions[tag][token]) / emissionsTotal[tag]))

# viterbi process for models
def viterbi(model, token):
    init_state = "init"
    final_state = "final"
    OOV_symbol = "OOV"
    verbose = False

    # Dictionaries to store transition and emission probabilities
    A = {}
    B = {}
    States = {}
    Voc = {}

    # load HMM model
    with open(model, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            if tokens[0] == "trans":
                qq, q, p = tokens[1], tokens[2], float(tokens[3])
                A.setdefault(qq, {})[q] = math.log(p)
                States[qq] = 1
                States[q] = 1
            elif tokens[0] == "emit":
                q, w, p = tokens[1], tokens[2], float(tokens[3])
                B.setdefault(q, {})[w] = math.log(p)
                States[q] = 1
                Voc[w] = 1
        #print(Voc)

    # Process each sentence
    out_path = 'btb.out'
    with open(out_path, 'a', encoding='utf-8') as file:
        with open(token, 'r', encoding='utf-8') as f:
            for line in f:
                w = line.strip().split()
                n = len(w)
                w.insert(0, "")
                V = {}
                Backtrace = {}
                V[0] = {init_state: 0.0}

                for i in range(1, n+1):
                    if w[i] not in Voc:
                        if verbose:
                            print(f"OOV: {w[i]}", file=sys.stderr)
                        w[i] = OOV_symbol
                    for q in States.keys():
                        for qq in States.keys():
                            if qq in A and q in A.get(qq, {}) and w[i] in B.get(q, {}) and qq in V.get(i-1, {}):
                                v = V[i-1][qq] + A[qq][q] + B[q][w[i]]
                                if i not in V or (q in V[i] and v > V[i][q]) or q not in V[i]:
                                    V.setdefault(i, {})[q] = v
                                    Backtrace.setdefault(i, {})[q] = qq
                                if verbose:
                                    print(f"V[{i}, {q}] = {V[i][q]} ({B[q][w[i]]})", file=sys.stderr)

                # Final state handling
                found_goal = False
                goal = float('-inf')
                for qq in States.keys():
                    if qq in A and final_state in A.get(qq, {}) and qq in V.get(n, {}):
                        v = V[n][qq] + A[qq][final_state]
                        if not found_goal or v > goal:
                            goal = v
                            found_goal = True
                            q = qq

                # Backtracking
                
                if found_goal:
                    t = []
                    for i in range(n, 0, -1):
                        t.insert(0, q)
                        q = Backtrace[i][q]
                    file.write(" ".join(t) + '\n')
                else:
                    file.write('\n')

# evaluation metrics
def evalaute_tag_acc(golds, hypos):
    with open(golds, encoding='utf-8') as goldFile, open(hypos, encoding='utf-8') as hypoFile:
        golds = goldFile.readlines()
        hypos = hypoFile.readlines()

        if len(golds) != len(hypos):
            raise ValueError("Length is different for two files!")
    tag_errors = 0
    sent_errors = 0
    tag_tot = 0
    sent_tot = 0

    for g, h in zip(golds, hypos):
        g = g.strip()
        h = h.strip()

        g_toks = re.split("\s+", g)
        h_toks = re.split("\s+", h)

        error_flag = False

        for i in range(len(g_toks)):
            if i >= len(h_toks) or g_toks[i] != h_toks[i]:
                tag_errors += 1
                error_flag = True

            tag_tot += 1

        if error_flag:
            sent_errors += 1

        sent_tot += 1

    error_word = tag_errors / tag_tot
    error_sentence = sent_errors / sent_tot
    print("error rate by word:      ", error_word, f" ({tag_errors} errors out of {tag_tot})")
    print("error rate by sentence:  ", error_sentence, f" ({sent_errors} errors out of {sent_tot})")
    
    return error_word, error_sentence


def trigram():
    vocab = {}
    OOV_WORD = "OOV"
    INIT_STATE = "init"
    FINAL_STATE = "final"

    emissions = {}
    transitions = {}
    transitionsTotal = defaultdict(int)
    emissionsTotal = defaultdict(int)

    with open(TAG_FILE, encoding='utf-8') as tagFile, open(TOKEN_FILE, encoding='utf-8') as tokenFile:
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
    with open('btb_tri.hmm', 'a' , encoding='utf-8') as file:
        for (prevprevtag, prevtag) in transitions:
            for tag in transitions[(prevprevtag, prevtag)]:
                file.write(f"trans {prevprevtag} {prevtag} {tag} {float(transitions[(prevprevtag, prevtag)][tag]) / transitionsTotal[(prevprevtag, prevtag)]}")
                file.write('\n')

        for tag in emissions:
            for token in emissions[tag]:
                file.write(f"emit {tag} {token} {float(emissions[tag][token]) / emissionsTotal[tag]}")
                file.write('\n')

def tri_viterbi():
    # Initialize
    init_state = "init"
    final_state = "final"
    OOV_symbol = "OOV"
    verbose = True

    # TRIGRAM Dictionaries for probabilities
    TrigramA = {}
    B = {}
    States = {}
    Voc = {}

    # BIGRAM Dictionaries for probs
    bi_A = {}
    bi_B = {}
    bi_States = {}
    bi_Voc = {}


    # Load HMM model
    TRIGRAM_HMM_FILE = 'btb_tri.hmm'
    BIGRAM_HMM_FILE = 'btb.hmm'

    # load BIGRAM-HMM model
    with open(BIGRAM_HMM_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            if tokens[0] == "trans":
                qq, q, p = tokens[1], tokens[2], float(tokens[3])
                bi_A.setdefault(qq, {})[q] = math.log(p)
                bi_States[qq] = 1
                bi_States[q] = 1
            elif tokens[0] == "emit":
                q, w, p = tokens[1], tokens[2], float(tokens[3])
                bi_B.setdefault(q, {})[w] = math.log(p)
                bi_States[q] = 1
                bi_Voc[w] = 1


    # load TRIGRAM-HMM model
    with open(TRIGRAM_HMM_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            if tokens[0] == "trans":
                qq, q, q_next, p = tokens[1], tokens[2], tokens[3], float(tokens[4])
                TrigramA[(qq, q, q_next)] = math.log(p)
                States[qq] = 1
                States[q] = 1
                States[q_next] = 1
            elif tokens[0] == "emit":
                q, w, p = tokens[1], tokens[2], float(tokens[3])
                B.setdefault(q, {})[w] = math.log(p)
                States[q] = 1
                Voc[w] = 1

    with open('btb_tri.out', 'a',  encoding='utf-8') as file:
    # Process each sentence
        with open('data/btb.test.txt', 'r', encoding='utf-8') as f:
            for line in f:
                w = line.strip().split()
                n = len(w)
                w.insert(0, "")
                # Initialize Viterbi and backtrace matrices
                V = {}
                Backtrace = {}
                V[0] = {(init_state, init_state): 0.0}

                for i in range(1, n+1):
                    word = w[i] if w[i] in Voc else OOV_symbol
                    for next_state in States.keys():
                        for qq, q in V[i-1].keys():
                            if (qq, q, next_state) in TrigramA:
                                new_prob = V[i-1][(qq, q)] + TrigramA[(qq, q, next_state)] + B.get(next_state, {}).get(word, float('-inf'))
                                # print((qq, q, next_state), V)
                                if (q, next_state) not in V.get(i, {}) or new_prob > V.get(i, {}).get((q, next_state), float('-inf')):
                                    V.setdefault(i, {})[(q, next_state)] = new_prob
                                    Backtrace.setdefault(i, {})[(q, next_state)] = (qq, q)

                # Final state handling (adapted for trigram model)
                best_final_prob = float('-inf')
                best_final_states = None
                for qq, q in V[n].keys():
                    if (qq, q, final_state) in TrigramA:
                        final_prob = V[n][(qq, q)] + TrigramA[(qq, q, final_state)]
                        if final_prob > best_final_prob:
                            best_final_prob = final_prob
                            best_final_states = (qq, q)
                # print(Backtrace)
                # Backtrack (adapted for trigram model)
                if best_final_states:
                    t = []
                    qq, q = best_final_states
                    for i in range(n, 0, -1):
                        t.insert(0, q)
                        
                        qq, q = Backtrace[i][(qq, q)]
                    file.write(" ".join(t) + '\n')
                else:
                    # best sentence not found, backoff to bigram
                    #print()
                    V = {}
                    Backtrace = {}
                    V[0] = {init_state: 0.0}

                    for i in range(1, n+1):
                        if w[i] not in bi_Voc:
                            w[i] = OOV_symbol
                        for q in bi_States.keys():
                            for qq in bi_States.keys():
                                if qq in bi_A and q in bi_A.get(qq, {}) and w[i] in bi_B.get(q, {}) and qq in V.get(i-1, {}):
                                    v = V[i-1][qq] + bi_A[qq][q] + bi_B[q][w[i]]
                                    if i not in V or (q in V[i] and v > V[i][q]) or q not in V[i]:
                                        V.setdefault(i, {})[q] = v
                                        Backtrace.setdefault(i, {})[q] = qq
                                    
                    # Final state handling
                    found_goal = False
                    goal = float('-inf')
                    for qq in bi_States.keys():
                        if qq in bi_A and final_state in bi_A.get(qq, {}) and qq in V.get(n, {}):
                            v = V[n][qq] + bi_A[qq][final_state]
                            if not found_goal or v > goal:
                                goal = v
                                found_goal = True
                                q = qq

                    # Backtracking
                    if found_goal:
                        t = []
                        for i in range(n, 0, -1):
                            t.insert(0, q)
                            q = Backtrace[i][q]
                        file.write(" ".join(t) + '\n')
                    else:
                        file.write('\n')
                    

if __name__ == "__main__":
    TAG_FILE = 'data/btb.train.tgs'
    TOKEN_FILE = 'data/btb.train.txt'
    
    # training bigram HMM
    train_HMM()
    print('producing output from model...')
    model_path = 'btb.hmm'
    # evaluation
    viterbi(model_path, 'data/btb.test.txt')
    print('evaluating model...')
    out_path = 'btb.out'
    # test the error rates
    error_word, error_sentence = evalaute_tag_acc('data/btb.test.tgs', out_path)

    # train trigram HMM
    print('training trigram btb...')
    trigram()
    model_path = 'btb_tri.hmm'
    print('trigram viterbi...')
    tri_viterbi()
    out_path = 'btb_tri.out'
    error_word, error_sentence = evalaute_tag_acc('data/btb.test.tgs', out_path)

