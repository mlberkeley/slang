import yaml
import numpy as np

from dsgan import DSGAN
from story_parse import Parser
from vsem import VSEM

with open('fivesent_dsgan.yaml', 'r') as f:
    params = yaml.safe_load(f)

parse = params['parser']
sent = Parser(parse['datasets'], parse['filetype'], 20000, parse['seq_len'])
vsem = VSEM(params['vsem'])
dsgan = DSGAN(params['dsgan'])

def generate_story():
    story = vsem.decode_batch(dsgan.generate())
    for i in range(parse['num_sent']):
        print(sent.index_sentence_to_sentence(story[i]))
    print()

def main():
    if params['load']:
        pass
    else:
        train = params['training']
        for i in range(train['train_steps']):
            idx = i + 1
            if idx % train['print_every'] == 0:
                print("Training step {}".format(idx))
                generate_story()
            write_summaries = True if idx % train['write_every'] == 0 else False
            mus, lvs, _ = vsem.encode_batch(np.array(sent.get_random_index_story()))
            dsgan.train_discriminator(np.array([mus, lvs]))
            mus, lvs, _ = vsem.encode_batch(np.array(sent.get_random_index_story()))
            dsgan.train_generator(np.array([mus, lvs]), idx, write_summaries=write_summaries)

