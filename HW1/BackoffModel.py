import math, collections

class BackoffModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.bigramCounts = collections.defaultdict(lambda: 0)
    self.unigramCounts = collections.defaultdict(lambda: 0)
    self.vocabSize = 0
    self.total = 0
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """  
    # TODO your code here
    # Tip: To get words from the corpus, try
    #    for sentence in corpus.corpus:
    #       for datum in sentence.data:  
    #         word = datum.word
    for sentence in corpus.corpus:
      p_word = None
      for datum in sentence.data:
        word = datum.word
        self.unigramCounts[word] += 1
        if p_word:
          self.bigramCounts[(p_word, word)] += 1
        p_word = word
        self.total += 1
    self.vocabSize = len(self.unigramCounts)
    self.total += self.vocabSize

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    # TODO your code here
    score = 0.0
    p_word = None
    for word in sentence:
      if (p_word and self.bigramCounts[(p_word, word)] > 0):
        bi_count = self.bigramCounts[(p_word, word)]
        uni_count = self.unigramCounts[p_word]
        score += math.log(bi_count) - math.log(uni_count)
      else:
        uni_count = self.unigramCounts[word]
        score += math.log(uni_count + 1) - math.log(self.total)
      p_word = word
    #print(score)
    return score
