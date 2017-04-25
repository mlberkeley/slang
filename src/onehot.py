import numpy as np
import nltk, pprint
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
from urllib.request import urlopen
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
import re
import time
import pickle
import unicodedata
import matplotlib.pyplot as plt
from pylab import rcParams
import csv

class data:

    def cleanData(self, bookURLs):
        wordTokens = []
        sentTokens = []
        lst = []
        for book in bookURLs:
            print("Reading book: " + book)
            rawbook = ''
            if (self.textOrUrl == "url"):
                response = urlopen(book)
                rawbook = response.read().decode('utf8')
            elif (self.textOrUrl == "text"):
                with open(book, 'r') as myfile:
                    rawbook = myfile.read().replace('\n', '')
            elif (self.textOrUrl == "csv"):
                with open(book, 'r') as f:
                    reader = csv.reader(f)
                    lst = list(reader)
                    combined = [x[2:] for x in lst]
                    for i in range(len(combined)):
                        rawbook += combined[i][0]
                        rawbook += ' '
                        rawbook += combined[i][1]
                        rawbook += ' '
                        rawbook += combined[i][2]
                        rawbook += ' '
                        rawbook += combined[i][3]
                        rawbook += ' '
                        rawbook += combined[i][4]
                        rawbook += ' '
            else:
                raise ValueError("Invalid file type. Must be 'csv', 'url', or 'text'.")

            rawbook =  re.sub(r'\r*\n', " ", rawbook)
            rawbook = re.sub(r' +', " ", rawbook)
            rawbook = rawbook.replace('”','"').replace('“', '"').replace('``', '').replace("''", '').replace('" "', '')
            rawbook = rawbook.replace('chapter', '').replace('Chapter', '').replace('-LCB-', '').replace('-RCB-', '')
            rawbook = rawbook.replace('-LSB-', '').replace('-RSB', '').replace('-LRB', '').replace('-RRB-', '')
            rawbook = rawbook.replace('"\n"', '').replace('Chapter heading picture :', '')
            rawbook = re.sub(r'"', '', rawbook)
            rawbook = rawbook.lower()
            wordTokens += word_tokenize(rawbook)
            sentTokens += sent_tokenize(rawbook)
        print("Cleaning sentences...")
#         for i in range(len(sentTokens)):
#             sentTokens[i] = word_tokenize(sentTokens[i])
        sentTokens = [word_tokenize(word) for word in sentTokens]
        print("Finished Cleaning")

        # Creating one-hot words
        wordTokens = [x.lower() for x in wordTokens] # makes all words lowercase
        allwords = wordTokens

        ###### keeps most frequent self.numWords # of words, rest are UNK #######
        unique, counts = np.unique(allwords, return_counts=True)
        wordsAndCounts = np.asarray((unique, counts)).T
        wordFreq = sorted(wordsAndCounts, key=lambda x: x[1].astype(float))
        ints = [[x[0].astype(str), x[1].astype(float)] for x in wordFreq]
        words = [x[0] for x in ints]
        counts = [x[1] for x in ints]
        words = list(reversed(words))
        counts = list(reversed(counts))
        wordTokens = words[:self.numWords]

        wordTokens.append('ppaadd')
        wordTokens.append('unk')
        wordTokens = list(set(wordTokens))
        wordTokens = sorted(wordTokens)
        wordlb = preprocessing.LabelBinarizer()
        wordEncoding = wordlb.fit_transform(wordTokens)

        print('Cleaning Data Complete')
        return wordlb, wordEncoding, wordTokens, sentTokens, allwords #### wordlb, encoding, tokens, ppaadd

    def createDicts(self):
        print('Creating Dictionaries...')
        encodeDict = {} #{'this' : 5} if the one hot is 00001000...
        decodeDict = {}
        for i in range(len(self.wordTokens)):
            word = self.wordTokens[i]
            index_of_1 = i
            encodeDict[word] = index_of_1
            decodeDict[index_of_1] = word
        print('Created dictionaries')
        return encodeDict, decodeDict

    def getAllSentences(self):
        print('Creating all sentences...')
        allSentences = []
        for sent in self.sentTokens:
            sentOfOneHotWords = []
            sentOfOneHotWords = [self.word_to_index(word) for word in sent]
            while len(sentOfOneHotWords) < (self.maxSentenceLength + 1):
                sentOfOneHotWords.append(self.word_to_index('ppaadd'))
            sentOfOneHotWords = sentOfOneHotWords[:(self.maxSentenceLength)]
            allSentences.append(sentOfOneHotWords)
        print('Done creating sentences')
        return allSentences

    def word_to_index(self, word):
        if word in self.encodeDict:
            return self.encodeDict[word]
        else:
            return self.encodeDict['unk']

    # maps index of the 1 to actual onehot encoding
    def index_to_onehot(self, index):
        onehot = np.append(np.append(np.zeros(index), 1), np.zeros(self.num_unique_words - (index+1)))
        onehot = onehot.reshape(1, len(onehot))
        return onehot

    def getSentence(self, sentenceIndex):
        if sentenceIndex > len(self.allSentences):
            raise ValueError("Sentence index is greater number of sentences in corpus")
        return(self.allSentences[sentenceIndex])

    # Returns sentence with all words in onehot
    def getOneHotSentence(self, sentenceIndex):
        sentence = self.getSentence(sentenceIndex)
        onehotSentence = []
        for i in range(len(sentence)):
            onehotSentence.append(self.getWord(sentenceIndex, i))
        return(onehotSentence)

    # Returns [[00...000]], a one-hot encoded word at specified sentence and word index in a nested array (for decoding)
    def getWord(self, sentenceIndex, wordIndex):
        if wordIndex > self.maxSentenceLength+1:
            raise ValueError("Word index is greater than max sentence length")
        word = self.allSentences[sentenceIndex][wordIndex]
        if type(word) != int:
            return word
        else:
            return(self.index_to_onehot(self.getSentence(sentenceIndex)[wordIndex])) # returns [[000...000]]

    # Decodes word at specified sentence and word indicies back into English
    def decode(self, sentenceIndex, wordIndex):
        word = self.getWord(sentenceIndex, wordIndex)
        if type(word) == str: # 'START', 'END', 'PAD'
            return word
        return(self.wordlb.inverse_transform(word)[0])

    def one_hot_to_word(self, onehot):
        return self.wordlb.inverse_transform(onehot)[0]

    def one_hot_sentence_to_sentence(self, sent):
        real = [self.one_hot_to_word(word) for word in sent]
#         for word in sent:
#             real.append(self.one_hot_to_word(word))
        return real

    # returns numSentences random sentences with words in onehot
    def getBatch(self, numSentences):
        batch = []
        for i in range(numSentences):
            rand = np.random.random_integers(len(self.allSentences))
            batch.append(self.getOneHotSentence(rand))
        return batch

    def __init__(self, bookURLs, textOrUrl, numWords, maxLength, encodeDict=None, decodeDict=None): #"text" or "url" for textOrUrl
        self.textOrUrl = textOrUrl
        self.numWords = numWords
        self.maxSentenceLength = maxLength
        self.bookURLs = bookURLs
        self.wordlb, self.wordEncoding, self.wordTokens, self.sentTokens, self.allWords = self.cleanData(self.bookURLs)
        self.encodeDict = {} #{'this' : 5} if the one hot is 00001000...
        self.decodeDict = {} # {5: 'this}
        if ((encodeDict != None) and (decodeDict != None)):
            self.encodeDict, self.decodeDict = encodeDict, decodeDict
        else:
            self.encodeDict, self.decodeDict = self.createDicts()
        self.num_unique_words = len(self.decodeDict)
        self.allSentences = self.getAllSentences()
        print("Words:", len(self.allWords))
        print("Unique words:", self.wordEncoding.shape[0])
        print("Sentences:", len(self.allSentences))
        print('Data initialized')
