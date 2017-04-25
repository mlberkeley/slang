import numpy as np

from vsem import VSEM
from onehot import data

params = { 'load':True,
           'load_idx':1,
           'dir':'../vsem_models/fivesent/model',
           'seq_len':25,
           'vocab_size':3702,
           'encode_hid':512,
           'latent_dims':256,
           'decode_hid':512,
           'learning_rate':1e-4,
           'keep_prob':0.5,
           'kl_alpha_rate':5e-6,
           'batch_size':50 }

sent = data(["../data/sentence5.csv"], "csv", params['vocab_size'] - 2, params['seq_len'])
vsem = VSEM(params)

def copy_sentence(params):
    seq = sent.getBatch(1)[0]
    pred = vsem.predict(seq)
    print("Original sentence: {}".format(sent.one_hot_sentence_to_sentence(seq)))
    print("Copied sentence: {}".format(sent.one_hot_sentence_to_sentence(pred)))

def telephone(params, repeat):
    seq = sent.getBatch(1)[0]
    print("Original sentence: {}".format(sent.one_hot_sentence_to_sentence(seq)))
    for i in range(repeat):
        seq = vsem.predict(seq)
        print("Copy {}: {}".format(i, sent.one_hot_sentence_to_sentence(seq)))

def main():
    if params['load']:
        copy_sentence(params)
    else:
        train_steps = 1000000
        for i in range(train_steps):
            idx = i + 1
            if idx % 10000 == 0:
                print("Training step {}".format(idx))
            write_summaries = True if idx % 10 == 0 else False
            vsem.train(sent.getBatch(params['batch_size']), idx, params, write_summaries=write_summaries)
            if idx % 100000 == 0:
                vsem.save(idx)

if __name__ == "__main__":
    main()
