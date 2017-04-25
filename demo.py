import numpy as np
import tensorflow as tf
from onehot import data

# BEFORE RUNNING THIS, RUN encode_text(["../data/sentence5.csv"], "csv", save_name="sentence5", save=True)

ENCODINGSDIR = "./"
ENCODINGSNAME = "sentence5"
PARAMS = { 'load':True,
           'load_idx':7,
           'dir':'./',
           'encode_hid':100,
           'latent_dims':100,
           'decode_hid':100,
           # Don't change anything below this line.
           'keep_prob':1,
           'batch_size':1,
           'kl_alpha_rate':1e-5,
           'learning_rate':1e-4}
TEXTSOURCES = ["./sentence5.csv"]
TEXTTYPE = "csv"
VOCABSIZE = 3700
SEQLEN = 15

def demo():
    text = data(TEXTSOURCES, TEXTTYPE, VOCABSIZE, SEQLEN)
    params['vocab_size'] = VOCABSIZE
    params['seq_len'] = SEQLEN
    vsem = VSEM(PARAMS)
    encodings = np.load(ENCODINGSDIR + ENCODINGSNAME + "_mu.npy")
    while True:
        print("Getting random sentence...")
        sentence_num = np.random.randint(0, len(encodings))
        true_sentence = text.getSentence(sentence_num)
        encoding = encodings[sentence_num]
        recovered_sentence = vsem.decode(encoding)

        true_sentence_string = one_hot_sentence_to_string(text, true_sentence)
        recovered_sentence_string = one_hot_sentence_to_string(text, recovered_sentence)

        print("Original or recovered:")
        if np.random.rand() > 0.5:
            print(true_sentence_string)
        else:
            print(recovered_sentence_string)

        input()

def one_hot_sentence_to_string(text, sentence):
    string = ""
    index = 0
    word = text.one_hot_to_word(sentence[index])
    while word != "ppaadd" and index < len(sentence):
        string += word + " "
        index += 1
        word = text.one_hot_to_word(sentence[index])
    return string

if __name__ == "__main__":
    demo()
