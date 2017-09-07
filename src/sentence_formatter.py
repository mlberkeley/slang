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
# nltk.download() # Might need this if tokenize doens't work

class SentenceFormatter: # initialize with data(["bookURL1", "bookURL2"...])
    
class data:
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
        
    def cleanData(self, bookURLs):
        wordTokens = ['ppaadd']
        sentTokens = []
        for book in bookURLs:
            print("Reading book: " + book)
            response = urlopen(book)
            rawbook = response.read().decode('utf8')
            rawbook =  re.sub(r'\r*\n', " ", rawbook)
            rawbook = re.sub(r' +', " ", rawbook)
            rawbook = rawbook.replace('”','"').replace('“', '"')
            rawbook = rawbook.replace('" "', ' Quote. Quote ')
            rawbook = rawbook.replace('"\n"', ' Quote. Quote ')
            rawbook = re.sub(r'"', ' Quote ', rawbook)
            rawbook = rawbook.lower()
            wordTokens += word_tokenize(rawbook)
            sentTokens += sent_tokenize(rawbook)
        print("Cleaning sentences...")
        for i in range(len(sentTokens)):
            sentTokens[i] = word_tokenize(sentTokens[i])
        print("Finished Cleaning")

        # Creating one-hot words
        wordTokens = [x.lower() for x in wordTokens] # makes all words lowercase
        wordTokens = list(set(wordTokens))
        wordTokens = sorted(wordTokens)
        wordlb = preprocessing.LabelBinarizer()
        wordEncoding = wordlb.fit_transform(wordTokens)
        print("Number of unique words:", wordEncoding.shape[0])
        print('Cleaning Data Complete')
        return wordlb, wordEncoding, wordTokens, sentTokens #### wordlb, encoding, tokens, ppaadd
    
    def createDicts(self):
        print('Creating Dictionaries...')
        encodeDict = {} #{'this' : 5} if the one hot is 00001000...
        decodeDict = {}
        print('Creating dicts')
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
            for word in sent:
                index = self.word_to_index(word)
                sentOfOneHotWords.append(index)
            while len(sentOfOneHotWords) < (self.maxSentenceLength + 1):
                sentOfOneHotWords.append(self.word_to_index('ppaadd'))
            sentOfOneHotWords = sentOfOneHotWords[:(self.maxSentenceLength)]
            allSentences.append(sentOfOneHotWords)
        print('Done creating sentences')
        return allSentences
              
    def word_to_index(self, word):
        return self.encodeDict[word]

    # maps index of the 1 to actual onehot encoding
    def index_to_onehot(self, index):
        onehot = np.append(np.append(np.zeros(index), 1), np.zeros(self.num_unique_words - (index+1)))
        onehot = onehot.reshape(1, len(onehot))
        return onehot 
    
    def getSentence(self, sentenceIndex):
        if sentenceIndex >= len(self.allSentences):
            raise ValueError("Sentence index is greater number of sentences in corpus")
        return(self.allSentences[sentenceIndex])

    # Returns sentence with all words in onehot
    def getOneHotSentence(self, sentenceIndex):
        sentence = self.getSentence(sentenceIndex)
        onehotSentence = []
        for i in range(len(sentence)):
            onehotSentence.append(self.getWord(sentenceIndex, i)[0,:])
        return(onehotSentence)

    # Returns [[00...000]], a one-hot encoded word at specified sentence and word index in a
    # nested array (for decoding)
    def getWord(self, sentenceIndex, wordIndex):
        if wordIndex > self.maxSentenceLength+1:
            raise ValueError("Word index is greater than max sentence length")
        word = self.allSentences[sentenceIndex][wordIndex]
        if type(word) != int:
            return word
        else:
            return(self.index_to_onehot(self.getSentence(sentenceIndex)[wordIndex])) # returns [[000...000]]


    # Returns an int representing index of 1 in one hot encoding
    def getIndexWord(self, sentenceIndex, wordIndex):
        if wordIndex > self.maxSentenceLength+1:
            raise ValueError("Word index is greater than max sentence length")
        word = self.allSentences[sentenceIndex][wordIndex]
        if type(word) != int:
            return word
        else:
            return(self.getSentence(sentenceIndex)[wordIndex]) 

    # Decodes word at specified sentence and word indicies back into English
    def decode(self, sentenceIndex, wordIndex):
        word = self.getWord(sentenceIndex, wordIndex)
        if type(word) == str: # 'START', 'END', 'PAD'
            return word
        return(self.wordlb.inverse_transform(word)[0])
    
    def one_hot_to_word(self, onehot):
        return self.wordlb.inverse_transform(onehot)[0]
    
    def one_hot_sentence_to_sentence(self, sent):
        real = []
        for word in sent:
            real.append(self.one_hot_to_word(np.expand_dims(word, axis=0)))
        return real
    
    # Converts sentence full of indices into words
    def index_sent_to_sent(self, sent):
        onehot = self.index_sent_to_one_hot(sent)
        return np.array(self.one_hot_sentence_to_sentence(onehot)).reshape((self.maxSentenceLength))

    # Returns numSentences random sentences with words in onehot
    def getBatch(self, numSentences):
        batch = []
        for i in range(numSentences):
            rand = np.random.random_integers(len(self.allSentences) - 1)
            batch.append(self.getOneHotSentence(rand))
        return np.array(batch)
    
    def __init__(self, bookURLs, encodeDict=None, decodeDict=None):
        self.maxSentenceLength = 50
        self.bookURLs = bookURLs
        self.wordlb, self.wordEncoding, self.wordTokens, self.sentTokens = self.cleanData(self.bookURLs)
        self.encodeDict = {} #{'this' : 5} if the one hot is 00001000...
        self.decodeDict = {} # {5: 'this}
        if ((encodeDict != None) and (decodeDict != None)):
            self.encodeDict, self.decodeDict = encodeDict, decodeDict
        else:
            self.encodeDict, self.decodeDict = self.createDicts()
        self.num_unique_words = len(self.decodeDict)
        self.allSentences = self.getAllSentences()
        print('Data initialized')
