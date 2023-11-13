#!/usr/bin/python

"""
Implement the Viterbi algorithm in Python (no tricks other than logmath!), given an
HMM, on sentences, and outputs the best state path.
Please check `viterbi.pl` for reference.

Usage:  python viterbi.py hmm-file < text > tags

special keywords:
 $init_state   (an HMM state) is the single, silent start state
 $final_state  (an HMM state) is the single, silent stop state
 $OOV_symbol   (an HMM symbol) is the out-of-vocabulary word
"""
import math
import sys

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
hmm_file = "my.hmm" 
with open(hmm_file, 'r') as f:
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
for line in sys.stdin:
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
        if verbose:
            print(math.exp(goal), file=sys.stderr)
        print(" ".join(t))
    else:
        print()
