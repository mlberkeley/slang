import numpy as np
import tensorflow as tf
from onehot import data
from vsem import VSEM

# BEFORE RUNNING THIS, RUN encode_text(["../data/sentence5.csv"], "csv", save_name="sentence5", save=True)

ENCODINGSDIR = "./"
ENCODINGSNAME = "sentence5"
TEXTSOURCES = ["../data/sentence5.csv"]
TEXTTYPE = "csv"
VOCABSIZE = 3700
SEQLEN = 15
PARAMS = { 'load':True,
           'load_idx':7,
           'dir':'./',
           'encode_hid':512,
           'latent_dims':256,
           'decode_hid':512,
           # Don't change anything below this line.
           'keep_prob':1.0,
           'batch_size':1,
           'kl_alpha_rate':1e-5,
           'learning_rate':1e-4,
           'vocab_size':VOCABSIZE + 2,
           'seq_len':SEQLEN}

def demo():
    text = data(TEXTSOURCES, TEXTTYPE, VOCABSIZE, SEQLEN)
    vsem = VSEM(PARAMS)
    encodings = np.load(ENCODINGSDIR + ENCODINGSNAME + "_mu.npy")
    while True:
        print("Getting random sentence...")
        sentence_num = np.random.randint(0, len(encodings))
        print("Sentence no:", sentence_num)

        true_sentence = []
        for word in text.getOneHotSentence(sentence_num):
            true_sentence.append(word[0])
        true_sentence = np.array(true_sentence, dtype=np.float32)

        encoding = encodings[sentence_num]
        recovered_sentence = vsem.decode(encoding)

        true_sentence_string = one_hot_sentence_to_string(text, true_sentence)
        recovered_sentence_string = one_hot_sentence_to_string(text, recovered_sentence)

        print("Original or recovered:")
        if np.random.rand() > 1:
            print(true_sentence_string)
            input()
            print("That was the original sentence.")
            print("Recovered was:")
            print(recovered_sentence_string)
        else:
            print(recovered_sentence_string)
            input()
            print("That was the recovered sentence.")
            print("Original was:")
            print(true_sentence_string)

        print("")

def one_hot_sentence_to_string(text, sentence):
    string = ""
    index = 0
    word = text.one_hot_to_word(np.expand_dims(sentence[index], axis=0))
    while word != "ppaadd" and index < len(sentence) - 1:
        string += word + " "
        index += 1
        word = text.one_hot_to_word(np.expand_dims(sentence[index], axis=0))
    return string

if __name__ == "__main__":
    demo()
