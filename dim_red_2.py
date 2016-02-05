#combines vectorization via tf-idf and svd into one script

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.stem.wordnet import WordNetLemmatizer

#reading in the crap, minor preprocessing
responses = []
lemmatizer = WordNetLemmatizer() #lemmatising so arrows -> arrow, hells -> hell, etc.
baddies = set(['Spectre', 'Phoenix\n', 'Io\n'])
with open('data') as f:
	for line in f:
		words = line.split(' ')
		if words[0] not in baddies: #first response is always name 
			line = ' '.join([lemmatizer.lemmatize(word.lower()) for word in words])
			responses.append(line)

#vectorizing, removing all tokens that either show up only once, or on more than 10% of the the hero responses 
vectorizer = TfidfVectorizer(stop_words = 'english', max_df = 0.1, min_df = 2)
tfidf_mat = vectorizer.fit_transform(responses)

#svd
svd = TruncatedSVD(n_components = 300, random_state = 1)
dim_red_mat = svd.fit_transform(tfidf_mat)
print(sum(svd.explained_variance_ratio_))

#getting the top words for each componenet--might not be very meaningful
def get_component_words(n_words, n_components, fitted_model, feature_names):
	'''a function to extract the top n_words from the top n_componenets of fitted_model, irregardless of sign. 
	requires a correspondng list of feature_names.  
	returns a list of tuples of (coefficient, feature).'''

	top_features = []
	for component in fitted_model.components_[0:n_components]:
		top_features.append([(component[i], feature_names[i]) for i in np.absolute(component).argsort()[::-1][0:n_words]])
	return(top_features)
print(get_component_words(n_words = 5, fitted_model = svd, n_components = 3, feature_names = vectorizer.get_feature_names()))

#plotting
hero_names = [line.split(' ')[0] for line in responses]
x, y = dim_red_mat[:, 1], dim_red_mat[:, 2] #most people toss out the first dimension, as its heavily correlated with document length
plt.scatter(x, y, linewidths = 0, alpha = 0)
for i, name in enumerate(hero_names):
	plt.annotate(name, (x[i], y[i]))
plt.xlabel('2nd LSA Dimension', fontsize = 16)
plt.ylabel('3rd LSA Dimension', fontsize = 16)
plt.show()

#focusing in on the big blob
plt.scatter(x, y, linewidths = 0, alpha = 0)
for i, name in enumerate(hero_names):
	plt.annotate(name, (x[i], y[i]))
plt.xlim(-0.2, 0.2)
plt.ylim(-0.2, 0.2)
plt.xlabel('2nd LSA Dimension', fontsize = 16)
plt.ylabel('3rd LSA Dimension', fontsize = 16)
plt.show()
