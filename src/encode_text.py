from onehot import data
import numpy as np
import tensorflow as tf
import model
from vsem import VSEM
from onehot import data

DEFAULT_PARAMS = { 'load':True,
                   'load_idx':7,
                   'dir':'../vsem_models/test',
                   'encode_hid':100,
                   'latent_dims':100,
                   'decode_hid':100,
                   # Don't change anything below this line.
                   'keep_prob':1,
                   'batch_size':1,
                   'kl_alpha_rate':1e-5,
                   'learning_rate':1e-4}

# encode a single text
def encode_text(bookURLs, format, seq_len, encodeDict, decodeDict, params=DEFAULT_PARAMS, save_name='', save=False):
    text = data(bookURLs, format, len(decodeDict), seq_len, encodeDist, decodeDict)
    textLen = len(text.allSentences)

    onehot_text = []
    for i in range(textLen):
        onehot_text.append(text.getOneHotSentence(i))
    onehot_text = np.array(onehot_text)

    params['vocab_size'] = len(decodeDict)
    params['seq_len'] = seq_len

    vsem = VSEM(params)

    book_mu, book_log_var = vsem.sess.run([mu, log_var], feed_dict = {x: onehot_text})

    if save:
        np.save(save_name + "_mu.npy", book_mu[0,:])
        np.save(save_name + "_log_var.npy", book_log_var[0,:])

    return book_mu[0,:], book_log_var[0,:]
