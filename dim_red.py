import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import json

execfile("vectorize.py")

# get a bag of word frequencies of words uttered by that hero
def get_bag_of_word(aline):
  global word_to_num
  n_words = len(word_to_num)
  splitted = aline.split(" ")

  hero = splitted[0].replace("\n","")
  hero_words = splitted[1:]

  bag = [1.0001 for i in range(n_words)]
  for hero_w in hero_words:
    if hero_w in word_to_num:
      bag[word_to_num[hero_w]] += 1.0

  bag = np.array(bag)

  # take a log to curb the outlyers
  bag = np.log(bag)
  # normalize
  print hero, sum(bag)
  bag = bag / (sum(bag) + 1)

#  bag = [0.0 for i in range(n_words)]
#  for hero_w in hero_words:
#    if hero_w in word_to_num:
#      bag[word_to_num[hero_w]] = 1.0
  
  return (hero, bag)

# transform a hero_words pair to a low-dim representation
def red_dim(hero_bags, pca):
  hero_names = [x[0] for x in hero_bags]
  hero_high_dim = np.array([x[1] for x in hero_bags])
  # times 100 to make the number a bit bigger so easier to read
  hero_low_dim = pca.transform(hero_high_dim)
  hero_low_dim = list(hero_low_dim)
  return zip(hero_names, hero_low_dim)

fd = open("data", "r")

# for hack we're removing spectre or phoenix or io, they say weird things
all_hero_bags = []
for aline in fd.readlines():
  aja = get_bag_of_word(aline)
  if aja[0] not in ["Spectre", "Phoenix", "Io"]:
    all_hero_bags.append(aja)

# np array version for all the original high-dim spaced bags
original_bag_pts = np.array([x[1] for x in all_hero_bags])
# fit a pca to the data, reducing dimensions
pca = PCA(n_components=2)
pca.fit(original_bag_pts)
# transform the entries to low dimension
low_dims = red_dim(all_hero_bags, pca)
# convert to a dictionary
data_dict = dict()
for name, value in low_dims:
  data_dict[name] = list(value)
# dump to json
data_json = open("data_vis.js", "w")
data_json.write("data_vis = "+json.dumps(data_dict)+"\n")


