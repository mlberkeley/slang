from onehot import data
import numpy as np
import tensorflow as tf
import model
from vsem import VSEM

DEFAULT_PARAMS = { 'load':True,
                   'load_idx':7,
                   'dir':'./',
                   'encode_hid':512,
                   'latent_dims':256,
                   'decode_hid':512,
                   # Don't change anything below this line.
                   'keep_prob':1,
                   'batch_size':1,
                   'kl_alpha_rate':1e-5,
                   'learning_rate':1e-4}
VOCAB_SIZE = 3700

# encode a single text
def encode_text(sources, format, seq_len=15, encodeDict=None, decodeDict=None, params=DEFAULT_PARAMS, save_name='', save=False):
    if decodeDict is not None:
        text = data(sources, format, len(decodeDict), seq_len, encodeDict, decodeDict)
    else:
        text = data(sources, format, VOCAB_SIZE, seq_len, encodeDict, decodeDict)

    textLen = len(text.allSentences)

    params['vocab_size'] = len(text.decodeDict)
    params['seq_len'] = seq_len

    vsem = VSEM(params)

    book_mu = np.empty((textLen, 256))
    book_log_var = np.empty((textLen, 256))

    for i in range(len(text.allSentences)): #textLen):
        sentence = []
        for word in text.getOneHotSentence(i):
            sentence.append(word[0])
        sentence = np.array(sentence, dtype=np.float32)

        book_mu[i], book_log_var[i], _ = vsem.encode(sentence)

    if save:
        np.save(save_name + "_mu.npy", book_mu)
        np.save(save_name + "_log_var.npy", book_log_var)

    return book_mu, book_log_var
