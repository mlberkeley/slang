import yaml
import numpy as np

from vsem import VSEM
from story_parse import Parser

with open('fivesent_vsem.yaml', 'r') as f:
    params = yaml.safe_load(f)

sent = Parser(["../data/sentence5.csv"], "csv", 20000, params['seq_len'])
vsem = VSEM(params)

def copy_sentence(params, low, high):
    seq = sent.get_batch(1, low, high)[0]
    pred = vsem.predict(seq)
    print("Original sentence: {}".format(sent.index_sentence_to_sentence(seq)))
    print("Copied sentence: {}".format(sent.index_sentence_to_sentence(pred)))

def cvx_comb(params):
    seqs = sent.get_batch(2, 0, len(sent.all_sentences))
    print("Sentence 1: {}".format(sent.index_sentence_to_sentence(seqs[0])))
    print("Sentence 2: {}".format(sent.index_sentence_to_sentence(seqs[1])))

    mu1, log_var1, _ = vsem.encode(seqs[0])
    mu2, log_var2, _ = vsem.encode(seqs[1])

    for i in range(6):
        lbda = i / 5.0
        mu = lbda*mu1 + (1-lbda)*mu2
        var = lbda*np.exp(log_var1) + (1-lbda)*np.exp(log_var2)
        z = np.random.normal(mu, np.sqrt(var), mu.shape)
        pred = vsem.decode(z)
        print("At lambda {}: {}".format(lbda, sent.index_sentence_to_sentence(pred)))

def telephone(params, repeat):
    seq = sent.get_batch(1)[0]
    print("Original sentence: {}".format(sent.index_sentence_to_sentence(seq)))
    for i in range(repeat):
        seq = vsem.predict(seq)
        print("Copy {}: {}".format(i, sent.index_sentence_to_sentence(seq)))

def demo(params):
    while(True):
        seq = sent.get_batch(1)[0]
        if np.random.rand() > 0.5:
            pred = vsem.predict(seq)
            print(sent.index_sentence_to_sentence(pred))
            input()
            print("Generated, original sentence:")
            print(sent.index_sentence_to_sentence(seq))
        else:
            print(sent.index_sentence_to_sentence(seq))
            input()
            print("Original")
        print()

def main():
    if params['load']:
        copy_sentence(params, 0.9*len(sent.all_sentences), len(sent.all_sentences))
    else:
        train_set_idx = int(0.9*(len(sent.all_sentences)-1))
        train_steps = params['train_steps']
        for i in range(train_steps):
            idx = i + 1
            if idx % 10000 == 0:
                print("Training step {}".format(idx))
            write_summaries = True if idx % 10 == 0 else False
            vsem.train(sent.get_batch(params['batch_size'], 0, train_set_idx),
                       idx, write_summaries=write_summaries)
            if idx % 100000 == 0:
                vsem.save(idx)

if __name__ == "__main__":
    main()
