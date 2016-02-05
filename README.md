# dota_response_cluster
a clustering of dota heros based on their responses

# files

## scraper.py
running this will scrape for the data file online, but since data is already here no need to run it again

## data
the raw data files from scraping the web, each line of the form: hero-name, word1, word2, ...

## vectorize.py
code to convert a word to a vector encoding

## dim_red.py
the dimensionality reduction

running this will generate data_vis.js

there are some heuristics in vectorize.py and dim_red.py that can change the clustering, you can mess with those

## data_vis.js
the visualization data encoded in json

i imagine you can take this and visualize it better, but i don't have time.

## index.html
a simple html5 canvas to visualize the data in data_vis.js
