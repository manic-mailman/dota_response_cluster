import operator
data = open("data", "r")

# grab all the word frequency
# measured by how many different heros say them
vocab_freq = dict()
total_hero_n = 0

for l in data.readlines():
  raw_words = l.split(" ")
  words = set(raw_words)
  for x in words:
    if x in vocab_freq:
      vocab_freq[x] += 1
    else: 
      vocab_freq[x] = 1
  total_hero_n += 1

sorted_vocab_freq = sorted(vocab_freq.items(), key=operator.itemgetter(1))

# get rid of useless words like "the" or "haha", which 
# many hero uses, this way we only track relevant words for similarity
vocab = set()
for x in vocab_freq:
  if vocab_freq[x] < total_hero_n / 2:
    vocab.add(x) 

# the mapping from word to number
vocab = list(vocab)
vocab.sort()
word_to_num = dict()
for i, wrd in enumerate(vocab):
  word_to_num[wrd] = i


