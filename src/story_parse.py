import numpy as np
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
from urllib.request import urlopen
import re
import unicodedata
import csv

PAD_WORD = 'ppaadd'
UNK = 'unk'

class Parser:

    """
    Read through the specified books and extract a cleaned dataset of stories and sentences

    :param book_urls: URLs or path to story files to parse
    :return: tuple of word tokens, sentences tokens, all words in vocabulary, and stories, all as
             lists containing the individual elements
    """
    def _clean(self, book_urls):
        word_tokens = []
        sent_tokens = []
        lst = []
        stories = []
        for book in book_urls:
            print("Reading book: " + book)
            rawbook = ''
            if (self.text_or_url == "url"):
                response = urlopen(book)
                rawbook = response.read().decode('utf8')
            elif (self.text_or_url == "text"):
                with open(book, 'r') as myfile:
                    rawbook = myfile.read().replace('\n', '')
            elif (self.text_or_url == "csv"):
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
                        stories.append(combined[i])
            else:
                raise ValueError("Invalid file type. Must be 'csv', 'url', or 'text'.")

            rawbook =  re.sub(r'\r*\n', " ", rawbook)
            rawbook = re.sub(r' +', " ", rawbook)
            rawbook = rawbook.replace('”','"').replace('“', '"').replace('``', '') \
                             .replace("''", '').replace('" "', '')
            rawbook = rawbook.replace('chapter', '').replace('Chapter', '').replace('-LCB-', '') \
                             .replace('-RCB-', '')
            rawbook = rawbook.replace('-LSB-', '').replace('-RSB', '').replace('-LRB', '') \
                             .replace('-RRB-', '')
            rawbook = rawbook.replace('"\n"', '').replace('Chapter heading picture :', '')
            rawbook = re.sub(r'"', '', rawbook)
            rawbook = rawbook.lower()
            word_tokens += word_tokenize(rawbook)
            sent_tokens += sent_tokenize(rawbook)
        print("Cleaning sentences...")
        sent_tokens = [word_tokenize(word) for word in sent_tokens]
        print("Finished Cleaning")

        # Creating one-hot words
        word_tokens = [x.lower() for x in word_tokens] # makes all words lowercase
        all_words = word_tokens

        ###### keeps most frequent self.num_words # of words, rest are UNK #######
        unique, counts = np.unique(all_words, return_counts=True)
        words_and_counts = np.asarray((unique, counts)).T
        word_freq = sorted(words_and_counts, key=lambda x: x[1].astype(float))
        ints = [[x[0].astype(str), x[1].astype(float)] for x in word_freq]
        words = [x[0] for x in ints]
        counts = [x[1] for x in ints]
        words = list(reversed(words))
        counts = list(reversed(counts))
        word_tokens = words[:self.num_words]

        word_tokens.append(PAD_WORD)
        word_tokens.append(UNK)
        word_tokens = list(set(word_tokens))
        word_tokens = sorted(word_tokens)

        print('Cleaning Data Complete')
        return word_tokens, sent_tokens, all_words, stories

    """
    Creates the encoding and decoding dictionaries, where the encoding dictionary contains the
    mapping from word to one-hot index, and the decoding dictionary contains the mapping from
    one-hot index to word
    
    :return: Tuple containing the encoding dictionary and decoding dictionary, in that order
    """
    def create_dicts(self):
        print('Creating Dictionaries...')
        encode_dict = {} #{'this' : 5} if the one hot is 00001000...
        decode_dict = {}
        for i in range(len(self.word_tokens)):
            word = self.word_tokens[i]
            index_of_1 = i
            encode_dict[word] = index_of_1
            decode_dict[index_of_1] = word
        print('Created dictionaries')
        return encode_dict, decode_dict

    """
    Collects all sentences parsed by the parser with words represented as one-hot indices

    :return: List containing all sentences, each of which is a list of one-hot indices
    """
    def _get_all_sentences(self):
        print('Creating all sentences...')
        all_sentences = []
        for sent in self.sent_tokens:
            sent_of_indices = [self.word_to_index(word) for word in sent]
            while len(sent_of_indices) <= (self.max_sentence_length):
                sent_of_indices.append(self.word_to_index(PAD_WORD))
            sent_of_indices = sent_of_indices[:(self.max_sentence_length)]
            all_sentences.append(sent_of_indices)
        print('Done creating sentences')
        return all_sentences

    """
    Collects all sentences parsed by the parser with each sentence in the story containing words
    represented as one-hot indices

    :return: List containing all sentences, each of which is a list of one-hot indices
    """
    def _get_all_stories(self):
        print('Collecting all stories...')
        all_stories = []
        for story in self.stories:
            all_stories.append([self.sentence_to_index_sentence(sent) for sent in story])
        print('Done collecting stories')
        return all_stories

    """
    Returns the encoded index of the word; if the word is not in the encoding dictionary, returns
    the unk index

    :param word: String word to be converted into an index
    :return: The index of the one-hot vector associated with the given word
    """
    def word_to_index(self, word):
        if word in self.encode_dict:
            return self.encode_dict[word]
        else:
            return self.encode_dict[UNK]

    """
    Returns the word that the given index encodes; if the index is not in the encoding dictionary,
    returns the 'UNK' word

    :param index: Integer index to be converted into a word
    :return: The word associated with the given one-hot vector index
    """
    def index_to_word(self, index):
        if index in self.decode_dict:
            return self.decode_dict[index]
        else:
            return UNK

    """
    Retrieves the specified sentence stored in the dataset

    :param sentence_index: The index of the sentence in the dataset to be retrieved
    :return: The specified sentence represented as a list of word indices
    """
    def get_sentence(self, sentence_index):
        if sentence_index >= len(self.all_sentences):
            raise ValueError("Sentence index is greater number of sentences in corpus")
        return self.all_sentences[sentence_index]

    """
    Retrieves the specified word stored in the dataset

    :param sentence_index: The index of the sentence that the word is contained in
    :param word_index: The index of the word within the given sentence
    :return: The specified word in one-hot vector index notation
    """
    def get_word(self, sentence_index, word_index):
        if word_index > self.max_sentence_length+1:
            raise ValueError("Word index is greater than max sentence length")
        return self.all_sentences[sentence_index][word_index]

    """
    Takes a sentence with words represented as indicies and converts it into a sentence with
    words as strings

    :param sent: Sentence represented as a list of integers
    :return: Sentence represented as a list of string words
    """
    def index_sentence_to_sentence(self, sent):
        return [self.index_to_word(index) for index in sent]

    """
    Converts a sentence with words as strings into the equivent sentence with index word
    representation

    :param sent: The sentence as a list of word strings to be converted into an index sentence
    :return: The sentence with words represented as indices, padded or clipped to the appropriate
             length
    """
    def sentence_to_index_sentence(self, sent):
        index_sent = [self.word_to_index(word) for word in word_tokenize(sent.lower())]
        while len(index_sent) <= self.max_sentence_length:
            index_sent.append(self.word_to_index(PAD_WORD))
        return index_sent[:self.max_sentence_length]

    """
    Creates a sentence filled with pad words

    :return: A sentence containing only the index representation of the pad word
    """
    def pad_sentence(self):
        return [self.word_to_index(PAD_WORD) for _ in range(self.max_sentence_length)]

    """
    Retrieves a random batch of sentences from the database

    :param num_sentences: Number of sentences to retrieve
    :param start_index: The earliest index of sentences to retrieve
    :param end_index: The latest index of sentences to retrieve
    :return: A batch of sentences in index representation
    """
    def get_batch(self, num_sentences, start_index, end_index):
        batch = []
        for i in range(num_sentences):
            rand = np.random.random_integers(start_index, end_index)
            batch.append(self.get_sentence(rand))
        return np.array(batch)

    """
    Retrieve a random story from the dataset

    :param threshold: The maximum fraction of the index to retrieve
    :return: A random story as a list containing sentences, each of which is represented as a list
             of words in indexed format
    """
    def get_random_index_story(self, threshold=1):
        marker = int(threshold*(len(self.all_stories)-1))
        rand = np.random.random_integers(marker)
        return self.all_stories[rand]

    """
    Retrieve a random story in human readable format

    :return: A random story presented as a single string with all sentences appended together
    """
    def get_random_story(self): 
        story = ''
        rand = np.random.random_integers(len(self.all_stories))
        for i in range(5):
            sent = self.index_sentence_to_sentence(self.all_stories[rand][i])
            text = ''
            j = 0
            while j < len(sent):
                word = sent[j]
                if word == PAD_WORD:
                    break
                text += word
                text += ' '
                j += 1
            text = text[:-2]
            text += "."
            story += text
            story += ' ' 
        return story

    def __init__(self, book_urls, text_or_url, num_words, max_length,
                       encode_dict=None, decode_dict=None):
        self.text_or_url = text_or_url
        self.num_words = num_words
        self.max_sentence_length = max_length
        self.book_urls = book_urls
        self.word_tokens, self.sent_tokens, self.all_words, self.stories = \
            self._clean(self.book_urls)
        self.encode_dict = {} #{'this' : 5} if the one hot is 00001000...
        self.decode_dict = {} # {5: 'this}
        if ((encode_dict != None) and (decode_dict != None)):
            self.encode_dict, self.decode_dict = encode_dict, decode_dict
        else:
            self.encode_dict, self.decode_dict = self.create_dicts()
        self.num_unique_words = len(self.decode_dict)
        self.all_sentences = self._get_all_sentences()
        self.all_stories = self._get_all_stories()
        print("Words:", len(self.all_words))
        print("Unique words:", len(self.encode_dict.keys()))
        print("Sentences:", len(self.all_sentences))
        print('Data initialized')
