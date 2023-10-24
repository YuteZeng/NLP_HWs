import math, collections
class CustomModel:
# reference KneserNey smoothing technique to improve performance based on smooth-bigram model
  def __init__(self, corpus, delta = 0.22):
    """Initial custom language model and structures needed by this mode"""
    self.bigramCounts = collections.defaultdict(lambda: 0)
    self.unigramCounts = collections.defaultdict(lambda: 0)
    self.following = collections.defaultdict(set)
    self.preceding = collections.defaultdict(set)
    self.totalBigrams = 0
    self.total = 0
    self.delta = delta
    self.train(corpus)


  def train(self, corpus):
    """ Takes a corpus and trains your language model.
    """  
    # TODO your code here
    
    for sentence in corpus.corpus:
      p_word = None  
      for datum in sentence.data:
        word = datum.word
        self.unigramCounts[word] += 1
        if p_word:
          self.bigramCounts[(p_word, word)] += 1
          self.following[p_word].add(word)
          self.preceding[word].add(p_word)
          self.totalBigrams += 1
        p_word = word
        self.total += 1
    self.vocabSize = len(self.unigramCounts)
    self.total += self.vocabSize

  def score(self, sentence):
    """ With list of strings, return the log-probability of the sentence with language model. Use
        information generated from train.
    """
    score = 0.0
    p_word = None
    for word in sentence:
      if p_word and self.bigramCounts[(p_word, word)] > 0:
        bi_count = self.bigramCounts[(p_word, word)]
        uni_count = self.unigramCounts[p_word]
        lambda_term = (self.delta / uni_count) * len(self.following[p_word])
        pk_word = len(self.preceding[word]) / self.totalBigrams
        knprob = max(bi_count - self.delta, 0) / uni_count + lambda_term * pk_word
        score += math.log(knprob)
      else:
        uni_count = self.unigramCounts[word]
        score += math.log(uni_count + 1) - math.log(self.total)
      p_word = word
    return score
