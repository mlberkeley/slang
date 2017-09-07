import numpy as np
import random

from bstm_v4 import BSTM
from story_parse import Parser

params = { 'load':False,
           'load_idx':0,
           'dir':'../bstm_models/v4/model',
           'seq_len':25,
           'vocab_size':20002,
           'word_dims':2048,
           'encode_hid':256,
           'latent_dims':256,
           'decode_hid':256,
           'learning_rate':1e-4,
           'keep_prob':0.5,
           'batch_size':50,
           'kl_alpha_rate':5e-6 }

parse = Parser(["../data/sentence5.csv"], "csv", params['vocab_size'] - 2, params['seq_len'])
bstm = BSTM(params, 0.6)

def copy_sentence(params, low, high):
    seq = parse.get_random_index_story(1)[random.randint(0, 4)]
    pred = bstm.predict(seq)
    print("Original sentence: {}".format(parse.index_sentence_to_sentence(seq)))
    print("Copied sentence: {}".format(parse.index_sentence_to_sentence(pred)))

def cvx_comb(params):
    seqs_0 = parse.get_random_index_story(1)[random.randint(0, 4)]
    seqs_1 = parse.get_random_index_story(1)[random.randint(0, 4)]
    print("Sentence 1: {}".format(parse.index_sentence_to_sentence(seqs_0)))
    print("Sentence 2: {}".format(parse.index_sentence_to_sentence(seqs_1)))

    _, _, z1 = bstm.encode(seqs_0)
    _, _, z2 = bstm.encode(seqs_1)

    for i in range(6):
        lbda = i / 5.0
        pred = bstm.decode(z1*lbda + z2*(1-ldba))
        print("At lambda {}: {}".format(parse.index_sentence_to_sentence(pred)))

def main():
    if not params['load']:
        train_steps = 2000000
        for i in range(train_steps):
            idx = i + 1
            if idx % 10000 == 0:
                print("Training step {}".format(idx))
            write_summaries = True if idx % 10 == 0 else False
            batch = []
            for i in range(params['batch_size']):
                story = parse.get_random_index_story(0.9)
                select = random.randint(0, 4)
                pre = parse.pad_sentence() if select == 0 else story[select - 1]
                post = parse.pad_sentence() if select == 4 else story[select + 1]
                batch.append([pre, story[select], post])
            batch = np.array(batch)
            bstm.train(batch, idx, write_summaries=write_summaries)
            if idx % 100000 == 0:
                bstm.save(idx)

if __name__ == "__main__":
    main()
