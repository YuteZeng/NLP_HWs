import math
import collections
from Corpus import Corpus

class Edit(object):
  """Holder object for edits (and the rules used to generate them)."""
  def __init__(self, editedWord, corruptLetters, correctLetters):
    self.editedWord = editedWord
    # Represents x in the "P(x|w)" error probability term of the noisy channel model
    self.corruptLetters = corruptLetters
    # Represents w in the "P(x|w)" error probability term of the noisy channel model
    self.correctLetters = correctLetters

  def rule(self):
    return "%s|%s" % (self.corruptLetters, self.correctLetters)

  def __hash__(self):
    return hash(str(self))

  def __eq__(self, o):
    return str(self) == str(o)

  def __str__(self):
    return "Edit(editedWord=%s, rule=%s)" % (self.editedWord, self.rule())

class EditModel(object):
  """An object representing the edit model for a spelling correction task."""

  ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
  def __init__(self, editFile="data/count_1edit.txt", corpus=None):
    if corpus:
      self.vocabulary = corpus.vocabulary()

    self.editCounts = {}
    with open(editFile, encoding ='ISO-8859-1') as f:
      for line in f:
        # import pdb
        # pdb.set_trace()
        rule, countString = line.split("\t")
        self.editCounts[rule] = int(countString)
    
  def deleteEdits(self, word):
    """Returns a list of edits of 1-delete distance words and rules used to generate them."""
    if len(word) <= 0:
      return []

    word = "<" + word #Append start character
    ret = []
    for i in range(1, len(word)):
      #The corrupted signal are this character and the character preceding
      corruptLetters = word[i-1:i+1] 
      #The correct signal is just the preceding character
      correctLetters = corruptLetters[:-1]

      #The corrected word deletes character i (and lacks the start symbol)
      correction = "%s%s" % (word[1:i], word[i+1:])
      ret.append(Edit(correction, corruptLetters, correctLetters))
      
    return ret

  def insertEdits(self, word):
    """Returns a list of edits of 1-insert distance words and rules used to generate them."""
    # TODO: write this
    # Tip: you might find EditModel.ALPHABET helpful
    # Tip: If inserting the letter 'a' as the second character in the word 'test', the corrupt
    #      signal is 't' and the correct signal is 'ta'. 
    if len(word) <= 0:
      return []
    word = "<" + word # append start token
    ret = []
    # there are len(word) + 1 insertion possible
    for i in range(0, len(word)):
      corruptLetters = word[i]
      # add one alpha in the alphabet one at a time
      for alpha in EditModel.ALPHABET:
        correctLetters = corruptLetters + alpha
        correction = "%s%s%s" % (word[1:i+1], alpha, word[i+1:])
        ret.append(Edit(correction, corruptLetters, correctLetters))

    return ret

  def transposeEdits(self, word):
    """Returns a list of edits of 1-transpose distance words and rules used to generate them."""
    # TODO: write this
    # Tip: If tranposing letters 'te' in the word 'test', the corrupt signal is 'te'
    #      and the correct signal is 'et'. 
    if len(word) <= 0:
      return []
    
    if len(word) == 1:
      return []
    word = "<" + word #Append start character
    ret = []
    for i in range(1, len(word)-1):
      if word[i] == word[i+1]:
        continue
      else:
        #The corrupted signal are this character and the character acceding
        corruptLetters = word[i:i+2] 
        #The correct signal is the transposed characters
        correctLetters = word[i+1:i-1:-1]

        #The corrected word deletes character i (and lacks the start symbol)
        correction = "%s%s%s" % (word[1:i], correctLetters, word[i+2:])
        ret.append(Edit(correction, corruptLetters, correctLetters))
      
    return ret

  def replaceEdits(self, word):
    """Returns a list of edits of 1-replace distance words and rules used to generate them."""
    # TODO: write this
    # Tip: you might find EditModel.ALPHABET helpful
    # Tip: If replacing the letter 'e' with 'q' in the word 'test', the corrupt signal is 'e'
    #      and the correct signal is 'q'.
    if len(word) <= 0:
      return []
    
    word = "<" + word
    ret = []
    for i in range(1, len(word)):
      corruptLetters = word[i]
      for alpha in EditModel.ALPHABET:
        if word[i] == alpha:
          continue
        else:
          correctLetters = alpha
          correction = "%s%s%s" % (word[1:i], alpha, word[i+1:])
          ret.append(Edit(correction, corruptLetters, correctLetters))
    return ret

  def edits(self, word):
    """Returns a list of tuples of 1-edit distance words and rules used to generate them, e.g. ("test", "te|et")"""
    #Note: this is just a suggested implementation, feel free to modify it for efficiency
    return  self.deleteEdits(word) + \
      self.insertEdits(word) + \
      self.transposeEdits(word) + \
      self.replaceEdits(word)

  def editProbabilities(self, misspelling):
    """Computes in-vocabulary edits and edit-probabilities for a given misspelling.
       Returns list of (correction, log(p(mispelling|correction))) pairs."""

    wordCounts = collections.defaultdict(int)
    wordTotal  = 0
    for edit in self.edits(misspelling):
      if edit.editedWord != misspelling and edit.editedWord in self.vocabulary and edit.rule() in self.editCounts:
        ruleMass = self.editCounts[edit.rule()] 
        wordTotal += ruleMass
        wordCounts[edit.editedWord] += ruleMass

    #Normalize by wordTotal to make probabilities
    return [(word, math.log(float(mass) / wordTotal)) for word, mass in list(wordCounts.items())]

# Start: Sanity checking code.

def checkOverlap(edits, gold):
  """Checks / prints the overlap between a guess and gold set."""
  percentage = 100 * float(len(edits & gold)) / len(gold)
  missing = gold - edits
  extra = edits - gold
  print(("\tOverlap: %s%%" % percentage))
  print(("\tMissing edits: %s" % list(map(str, missing))))
  print(("\tExtra edits: %s" % list(map(str, extra))))

def main():
  """Sanity checks the edit model on the word 'hi'."""

  trainPath = 'data/tagged-train.dat'
  trainingCorpus = Corpus(trainPath)
  editModel = EditModel("data/count_1edit.txt", trainingCorpus)
  #These are for testing, you can ignore them
  DELETE_EDITS = set(['Edit(editedWord=i, rule=<h|<)', 'Edit(editedWord=h, rule=hi|h)'])
  INSERT_EDITS = set([Edit('ahi','<','<a'),Edit('bhi','<','<b'),Edit('chi','<','<c'),Edit('dhi','<','<d'),Edit('ehi','<','<e'),Edit('fhi','<','<f'),Edit('ghi','<','<g'),Edit('hhi','<','<h'),Edit('ihi','<','<i'),Edit('jhi','<','<j'),Edit('khi','<','<k'),Edit('lhi','<','<l'),Edit('mhi','<','<m'),Edit('nhi','<','<n'),Edit('ohi','<','<o'),Edit('phi','<','<p'),Edit('qhi','<','<q'),
    Edit('rhi','<','<r'),Edit('shi','<','<s'),Edit('thi','<','<t'),Edit('uhi','<','<u'),Edit('vhi','<','<v'),Edit('whi','<','<w'),Edit('xhi','<','<x'),Edit('yhi','<','<y'),Edit('zhi','<','<z'),Edit('hai','h','ha'),Edit('hbi','h','hb'),Edit('hci','h','hc'),Edit('hdi','h','hd'),Edit('hei','h','he'),Edit('hfi','h','hf'),Edit('hgi','h','hg'),Edit('hhi','h','hh'),
    Edit('hii','h','hi'),Edit('hji','h','hj'),Edit('hki','h','hk'),Edit('hli','h','hl'),Edit('hmi','h','hm'),Edit('hni','h','hn'),Edit('hoi','h','ho'),Edit('hpi','h','hp'),Edit('hqi','h','hq'),Edit('hri','h','hr'),Edit('hsi','h','hs'),Edit('hti','h','ht'),Edit('hui','h','hu'),Edit('hvi','h','hv'),Edit('hwi','h','hw'),Edit('hxi','h','hx'),Edit('hyi','h','hy'),Edit('hzi','h','hz'),
    Edit('hia','i','ia'),Edit('hib','i','ib'),Edit('hic','i','ic'),Edit('hid','i','id'),Edit('hie','i','ie'),Edit('hif','i','if'),Edit('hig','i','ig'),Edit('hih','i','ih'),Edit('hii','i','ii'),Edit('hij','i','ij'),Edit('hik','i','ik'),Edit('hil','i','il'),Edit('him','i','im'),Edit('hin','i','in'),Edit('hio','i','io'),Edit('hip','i','ip'),Edit('hiq','i','iq'),Edit('hir','i','ir'),
    Edit('his','i','is'),Edit('hit','i','it'),Edit('hiu','i','iu'),Edit('hiv','i','iv'),Edit('hiw','i','iw'),Edit('hix','i','ix'),Edit('hiy','i','iy'),Edit('hiz','i','iz')])
  TRANPOSE_EDITS = set([Edit('ih','hi','ih')])
  REPLACE_EDITS = set([Edit('ai','h','a'),Edit('bi','h','b'),Edit('ci','h','c'),Edit('di','h','d'),Edit('ei','h','e'),Edit('fi','h','f'),Edit('gi','h','g'),Edit('ii','h','i'),Edit('ji','h','j'),
    Edit('ki','h','k'),Edit('li','h','l'),Edit('mi','h','m'),Edit('ni','h','n'),Edit('oi','h','o'),Edit('pi','h','p'),Edit('qi','h','q'),Edit('ri','h','r'),Edit('si','h','s'),Edit('ti','h','t'),
    Edit('ui','h','u'),Edit('vi','h','v'),Edit('wi','h','w'),Edit('xi','h','x'),Edit('yi','h','y'),Edit('zi','h','z'),Edit('ha','i','a'),Edit('hb','i','b'),Edit('hc','i','c'),Edit('hd','i','d'),Edit('he','i','e'),Edit('hf','i','f'),Edit('hg','i','g'),Edit('hh','i','h'),Edit('hj','i','j'),
    Edit('hk','i','k'),Edit('hl','i','l'),Edit('hm','i','m'),Edit('hn','i','n'),Edit('ho','i','o'),Edit('hp','i','p'),Edit('hq','i','q'),Edit('hr','i','r'),Edit('hs','i','s'),Edit('ht','i','t'),
    Edit('hu','i','u'),Edit('hv','i','v'),Edit('hw','i','w'),Edit('hx','i','x'),Edit('hy','i','y'),Edit('hz','i','z')])

  print("***Code Sanity Check***")
  print("Delete edits for 'hi'")
  checkOverlap(set(editModel.deleteEdits('hi')), DELETE_EDITS)
  print("Insert edits for 'hi'")
  checkOverlap(set(editModel.insertEdits('hi')), INSERT_EDITS)
  print("Transpose edits for 'hi'")
  checkOverlap(set(editModel.transposeEdits('hi')), TRANPOSE_EDITS)
  print("Replace edits for 'hi'")
  checkOverlap(set(editModel.replaceEdits('hi')), REPLACE_EDITS)

if __name__ == "__main__":
  main()
