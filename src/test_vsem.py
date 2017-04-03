import numpy as np

from vsem import VSEM

params = { 'load':True,
           'load_idx':7,
           'dir':'../vsem_models/test',
           'seq_len':30,
           'vocab_size':2000,
           'encode_hid':100,
           'latent_dims':100,
           'decode_hid':100,
           'learning_rate':1e-4,
           'keep_prob':0.5,
           'kl_alpha_rate':1e-5,
           'batch_size':50 }

vsem = VSEM(params)

def random_sequence(params):
    idx = np.random.zipf(1.25, params['seq_len'])
    seq = np.zeros((params['seq_len'], params['vocab_size']))
    sentence_end = False
    for i in range(idx.size):
        if sentence_end:
            seq[i, 0] = 1
        else:
            if i == 0 and idx[i] >= params['vocab_size']:
                seq[i, 1] = 1
            else:
                if idx[i] >= params['vocab_size']:
                    seq[i, 0] = 1
                    sentence_end = True
                else:
                    seq[i, idx[i]] = 1
    return seq

def sequence_batch(params):
    seq_batch = np.expand_dims(random_sequence(params), axis=0)
    for _ in range(params['batch_size']-1):
        seq_batch = np.concatenate((seq_batch, np.expand_dims(random_sequence(params), axis=0)), axis=0)
    return seq_batch

def predict_vsem(params):
    seq = random_sequence(params)
    pred = vsem.predict(seq)
    print("Original sequence: {}".format(np.argmax(seq, axis=1)))
    print("Predicted sequence: {}".format(np.argmax(pred, axis=1)))

def main():
    if params['load']:
        predict_vsem(params)
    else:
        train_steps = 1000000
        for i in range(train_steps):
            idx = i + 1
            if idx % 1000 == 0:
                print("Training step {}".format(idx))
            write_summaries = True if idx % 10 == 0 else False
            vsem.train(sequence_batch(params), idx, params, write_summaries=write_summaries)
            if idx % 100000 == 0:
                vsem.save(idx)

if __name__ == "__main__":
    main()
