import numpy as np

def ascii2one_hot(letter):
    vec = np.zeros(128)
    vec[ord(letter)] = 1
    return vec

def one_hot2ascii(vec):
    indices = np.flatnonzero(vec)
    return chr(indices[0])
