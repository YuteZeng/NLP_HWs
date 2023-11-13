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
def train_HMM(N):
    vocab = {}
    OOV_WORD = "OOV"
    INIT_STATE = "init"
    FINAL_STATE = "final"

    emissions = {}
    transitions = {}
    transitionsTotal = defaultdict(int)
    emissionsTotal = defaultdict(int)

    sentence_count = 0 

    with open(TAG_FILE) as tagFile, open(TOKEN_FILE) as tokenFile:
        for tagString, tokenString in zip(tagFile, tokenFile):
            sentence_count += 1
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

            if sentence_count >= N:
                break

    # model store
    model_path = 'split_training_HMMs/' + str(N) + '.hmm'
    with open(model_path, 'w') as file:
        for prevtag in transitions:
            for tag in transitions[prevtag]:
                print(("trans %s %s %s" % (prevtag, tag, float(transitions[prevtag][tag]) / transitionsTotal[prevtag])), file=file)

        for tag in emissions:
            for token in emissions[tag]:
                print(("emit %s %s %s " % (tag, token, float(emissions[tag][token]) / emissionsTotal[tag])), file=file)

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
    with open(model, 'r') as f:
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
    out_path = 'models_output/' + model.split('.')[0].split('/')[1] + '.out'
    with open(out_path, 'w') as file:
        with open(token, 'r') as f:
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
                    print(" ".join(t), file=file)
                else:
                    print(file=file)

# evaluation metrics
def evalaute_tag_acc(golds, hypos):
    with open(golds) as goldFile, open(hypos) as hypoFile:
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

# visualization in a figure
def visual(name, data):
    y_values = data
    # Generate corresponding x-axis values spaced evenly from 1000 to 40000
    x_values = np.linspace(1000, 40000, num=len(y_values))

    plt.figure(figsize=(10, 5))
    plt.plot(x_values, y_values, marker='o')

    # Label the axes
    plt.xlabel('Training Size')
    plt.ylabel(name)

    # Title of the graph
    title = name + ' vs Training Size'
    plt.title(title)

    # Save the figure to a PNG file
    output_path = 'figures/' + title.replace(' ', '_') + '.png'
    plt.savefig(output_path, dpi=300)
    plt.close()



if __name__ == "__main__":
    TAG_FILE = sys.argv[1]
    TOKEN_FILE = sys.argv[2]

    error_word_list = []
    error_sentence_list = []

    for size in range(1000, 41000, 2000):
        # training HMM
        print('training model on', size, 'sentences')
        train_HMM(size)
        print('producing output from model...')
        model_path = 'split_training_HMMs/' + str(size) + '.hmm'
        # evaluation
        viterbi(model_path, 'data/ptb.22.txt')
        print('evaluating model...')
        out_path = 'models_output/' + str(size) + '.out'
        # test the error rates
        error_word, error_sentence = evalaute_tag_acc('data/ptb.22.tgs', out_path)
        error_word_list.append(error_word)
        error_sentence_list.append(error_sentence)

    print(error_word_list, error_sentence_list)
    # visualization of learning curve
    visual('Error Rate by Word', error_word_list)
    visual('Error Rate by Sentence', error_sentence_list)