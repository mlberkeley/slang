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

    def getSentence(self, sentenceIndex):
        if sentenceIndex >= len(self.allSentences):
            raise ValueError("Sentence index is greater number of sentences in corpus")
        return(self.allSentences[sentenceIndex])

    # Returns sentence with all words in onehot
    def getOneHotSentence(self, sentenceIndex):
        sentence = self.getSentence(sentenceIndex)
        onehotSentence = []
        for i in range(len(sentence)):
            onehotSentence.append(self.getWord(sentenceIndex, i))
        return(onehotSentence)

    # Returns sentence with all words in as indices of 1 in onehot
    def getIndexSentence(self, sentenceIndex):
        sentence = self.getSentence(sentenceIndex)
        indexSentence = []
        for i in range(len(sentence)):
            indexSentence.append(self.getIndexWord(sentenceIndex, i))
        return(indexSentence)

    # Returns [[00...000]], a one-hot encoded word at specified sentence and word index in a nested array (for decoding)
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

    # Converts one hot sentence into english
    def one_hot_sentence_to_sentence(self, sent):
        sent = np.expand_dims(sent, axis=1)
        real = [self.one_hot_to_word(word) for word in sent]
        return np.array(real).reshape((self.maxSentenceLength))
    
    # Converts index of 1 in onehot into actual onehot
    def index_to_onehot(self, index):
        onehot = np.append(np.append(np.zeros(index), 1), np.zeros(self.num_unique_words - (index+1)))
        onehot = onehot.reshape(1, len(onehot))
        return onehot

    # Converts sentence full of indices into onehot
    def index_sent_to_one_hot(self, sent):
        sent = np.expand_dims(sent, axis=1)
        real = [self.index_to_onehot(index) for index in sent]
        real = [x[0] for x in real]
        return real
    
    # Converts sentence full of indices into words
    def index_sent_to_sent(self, sent):
        onehot = self.index_sent_to_one_hot(sent)
        return np.array(self.one_hot_sentence_to_sentence(onehot)).reshape((self.maxSentenceLength))

    # Returns numSentences random sentences with words in onehot
    def getBatch(self, numSentences):
        batch = []
        for i in range(numSentences):
            rand = np.random.random_integers(len(self.allSentences)-1)
            batch.append(self.getOneHotSentence(rand))
        return np.array(batch).reshape((numSentences, self.maxSentenceLength, self.numWords+2))

    # Returns numSentences random sentences with words as indices
    def getIndexBatch(self, numSentences):
        batch = []
        for i in range(numSentences):
            rand = np.random.random_integers(len(self.allSentences)-1)
            batch.append(self.getIndexSentence(rand))
        return np.array(batch).reshape((numSentences, self.maxSentenceLength))


    # returns a one hot of 5 sentences that corresponds to 1 story
    def getRandomStoryOneHot(self):
        batch = []
        rand = np.random.random_integers(len(self.allSentences))
        rand = rand - (rand % 5) - 1
        for i in range(rand, rand+5):
            batch.append(self.getOneHotSentence(i))
        return batch
    
    # returns a string of 5 sentences that corresponds to 1 story
    def getRandomStory(self): 
        batch = []
        story = ''
        rand = np.random.random_integers(len(self.allSentences))
        rand = rand - (rand % 5) - 1
        for i in range(rand, rand+5):
            sent = self.one_hot_sentence_to_sentence(np.array(self.getOneHotSentence(i)).reshape(self.maxSentenceLength, self.numWords+2))
            text = ''
            i = 0
            while i < len(sent):
                word = sent[i]
                if word == "ppaadd":
                    break
                text += word
                text += ' '
                i += 1
            text = text[:-2]
            text += "."
            story += text
            story += ' ' 
        return story