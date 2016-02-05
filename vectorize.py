import operator
data = open("data", "r")

# grab all the word frequency from the speech
vocab_freq = dict()
total = 0

for l in data.readlines():
  for x in l.split(" "):
    if x in vocab_freq:
      vocab_freq[x] += 1
    else: 
      vocab_freq[x] = 1
    total += 1

sorted_vocab_freq = sorted(vocab_freq.items(), key=operator.itemgetter(1))

# get only words that's been uttered more than once
vocab = set()
for x in vocab_freq:
  if vocab_freq[x] > 1:
    vocab.add(x) 

# the mapping from word to number
vocab = list(vocab)
vocab.sort()
word_to_num = dict()
for i, wrd in enumerate(vocab):
  word_to_num[wrd] = i


