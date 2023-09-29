import math, collections

class SmoothBigramModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.smoothbigramCounts = collections.defaultdict(lambda: 1)
    self.unigramCounts = collections.defaultdict(lambda: 0)
    self.vocabSize = 0
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
      # initialize the previous word
      p_word = None
      for datum in sentence.data:
        word = datum.word
        self.unigramCounts[word] += 1
        if p_word:
          self.smoothbigramCounts[(p_word, word)] += 1
        p_word = word
    # count vocabulary size
    # print(self.smoothbigramCounts)
    self.vocabSize = len(self.unigramCounts)

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    # TODO your code here
    score = 0.0
    p_word = None
    for word in sentence:
      if p_word:
        bi_count = self.smoothbigramCounts[(p_word, word)]
        uni_count = self.unigramCounts[p_word]
        score += math.log(bi_count) - math.log(uni_count + self.vocabSize)
      p_word = word
    return score
