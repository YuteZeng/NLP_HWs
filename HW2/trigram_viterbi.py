#!/usr/bin/python

"""
Implement the Viterbi algorithm in Python (no tricks other than logmath!), given an
HMM, on sentences, and outputs the best state path.
Please check `viterbi.pl` for reference.

Usage:  python viterbi.py trigram_hmm-file bigram_hmm_file < text > tags

special keywords:
 $init_state   (an HMM state) is the single, silent start state
 $final_state  (an HMM state) is the single, silent stop state
 $OOV_symbol   (an HMM symbol) is the out-of-vocabulary word
"""
import math
import sys

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
TRIGRAM_HMM_FILE = sys.argv[1]
BIGRAM_HMM_FILE = sys.argv[2]


# load BIGRAM-HMM model
with open(BIGRAM_HMM_FILE, 'r') as f:
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
with open(TRIGRAM_HMM_FILE, 'r') as f:
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


# Process each sentence
for line in sys.stdin:
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
            try:
                qq, q = Backtrace[i][(qq, q)]
            except KeyError:
                print(f"KeyError: Backtrace for time {i} and states (q={q}, next_state={next_state}) not found.")
                break  # Exit the loop

        print(" ".join(t))
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
            print(" ".join(t))
        else:
            print()
            