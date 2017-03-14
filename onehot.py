import urllib
import nltk, re, pprint
from nltk import word_tokenize
url = "http://www.gutenberg.org/files/2554/2554-0.txt"
print("at response")
response = urllib.urlopen(url)
print("at raw")
raw = response.read().decode('utf8')
print("tokenizing")
tokens = word_tokenize(raw)
tokens[:20]
