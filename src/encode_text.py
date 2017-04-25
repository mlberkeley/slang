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

# encode a single text
def encode_text(sources, format, seq_len=15, encodeDict=None, decodeDict=None, params=DEFAULT_PARAMS, save_name='', save=False):
    if decodeDict is not None:
        text = data(sources, format, len(decodeDict), seq_len, encodeDict, decodeDict)
    else:
        text = data(sources, format, 3700, seq_len, encodeDict, decodeDict)

    textLen = len(text.allSentences)

    onehot_text = []
    for i in range(5): #textLen):
        sentence = []
        for word in text.getOneHotSentence(i):
            sentence.append(np.array(word[0], dtype=np.float32))
        onehot_text.append(np.array(sentence, dtype=np.float32))
    onehot_text = np.array(onehot_text, dtype=np.float32)

    params['vocab_size'] = len(text.decodeDict)
    params['seq_len'] = seq_len

    vsem = VSEM(params)

    book_mu, book_log_var = vsem.sess.run([vsem.mu, vsem.log_var], feed_dict = {vsem.x:onehot_text,
                                                                                vsem.keep_prob:1.0,
                                                                                vsem.batch_size:1})

    if save:
        np.save(save_name + "_mu.npy", book_mu[0,:])
        np.save(save_name + "_log_var.npy", book_log_var[0,:])

    return book_mu[0,:], book_log_var[0,:]
