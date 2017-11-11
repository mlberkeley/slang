import numpy as np
import random
import os
import pickle

from story_parse import Parser
from gradcheck import gradcheck_naive

def normalizeRows(x):
    """ Row normalization function """
    # Implement a function that normalizes each row of a matrix to have unit length
    
    ### YOUR CODE HERE
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    ### END YOUR CODE
    
    return x

def test_normalize_rows():
    print("Testing normalizeRows...")
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]])) 
    # the result should be [[0.6, 0.8], [0.4472, 0.8944]]
    print(x)
    assert (x.all() == np.array([[0.6, 0.8], [0.4472, 0.8944]]).all())
    print("")

def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models """
    
    # Implement the cost and gradients for one predicted word vector  
    # and one target word vector as a building block for word2vec     
    # models, assuming the softmax prediction function and cross      
    # entropy loss.                                                   
    
    # Inputs:                                                         
    # - predicted: numpy ndarray, predicted word vector (\hat{v} in 
    #   the written component or \hat{r} in an earlier version)
    # - target: integer, the index of the target word               
    # - outputVectors: "output" vectors (as rows) for all tokens     
    # - dataset: needed for negative sampling, unused here.         
    
    # Outputs:                                                        
    # - cost: cross entropy cost for the softmax word prediction    
    # - gradPred: the gradient with respect to the predicted word   
    #        vector                                                
    # - grad: the gradient with respect to all the other word        
    #        vectors                                               
    
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!                                                  
    
    ### YOUR CODE HERE
    V_size, D = outputVectors.shape
    U = outputVectors

    o = np.zeros((V_size,1))
    o[target] = 1

    dx = (softmax(U.dot(predicted.reshape(D, 1))) - o)
    dv_c = dx.T.dot(U)
    dU = dx.dot(predicted.reshape(1,D))

    cost = -np.log(U.dot(predicted.reshape(D,1))[target])
    gradPred = dv_c
    grad = dU
    ### END YOUR CODE
    
    return cost, gradPred, grad

def negSamplingCostAndGradient(predicted, target, outputVectors, dataset, 
    K=10):
    """ Negative sampling cost function for word2vec models """

    # Implement the cost and gradients for one predicted word vector  
    # and one target word vector as a building block for word2vec     
    # models, using the negative sampling technique. K is the sample  
    # size. You might want to use dataset.sampleTokenIdx() to sample  
    # a random word index. 
    # 
    # Note: See test_word2vec below for dataset's initialization.
    #                                       
    # Input/Output Specifications: same as softmaxCostAndGradient     
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!
    
    ### YOUR CODE HERE
    sampleIdxs = [dataset.sampleTokenIdx() for _ in range(K)]
    cost = -np.log(sigmoid(outputVectors[target].dot(predicted)))
    gradPred = -sigmoid(-outputVectors[target].dot(predicted)) * outputVectors[target]
    grad = np.zeros_like(outputVectors)
    grad[target] = -sigmoid(-outputVectors[target].dot(predicted)) * predicted

    for sampleIdx in sampleIdxs:
        cost += -np.log(sigmoid(-outputVectors[sampleIdx].dot(predicted)))
        gradPred += sigmoid(outputVectors[sampleIdx].dot(predicted)) * outputVectors[sampleIdx]
        grad[sampleIdx] += sigmoid(outputVectors[sampleIdx].dot(predicted)) * predicted
    ### END YOUR CODE
    
    return cost, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors, 
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    """ Skip-gram model in word2vec """

    # Implement the skip-gram model in this function.

    # Inputs:                                                         
    # - currrentWord: a string of the current center word           
    # - C: integer, context size                                    
    # - contextWords: list of no more than 2*C strings, the context words                                               
    # - tokens: a dictionary that maps words to their indices in    
    #      the word vector list                                
    # - inputVectors: "input" word vectors (as rows) for all tokens           
    # - outputVectors: "output" word vectors (as rows) for all tokens         
    # - word2vecCostAndGradient: the cost and gradient function for 
    #      a prediction vector given the target word vectors,  
    #      could be one of the two cost functions you          
    #      implemented above

    # Outputs:                                                        
    # - cost: the cost function value for the skip-gram model       
    # - grad: the gradient with respect to the word vectors         
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!

    ### YOUR CODE HERE
    cost = 0
    gradIn = np.zeros_like(inputVectors)
    gradOut = np.zeros_like(outputVectors)
    for targetWord in contextWords:
        predicted = inputVectors[tokens[targetWord],:]
        target = tokens[targetWord]
        cost_t, gradPred, grad = word2vecCostAndGradient(predicted, target, outputVectors, dataset)
    
        cost += cost_t
        gradIn[target] += gradPred.reshape(-1)
        gradOut += grad
    ### END YOUR CODE
    
    return cost, gradIn, gradOut

def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors, 
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    """ CBOW model in word2vec """

    # Implement the continuous bag-of-words model in this function.            
    # Input/Output specifications: same as the skip-gram model        
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!

    #################################################################
    # IMPLEMENTING CBOW IS EXTRA CREDIT, DERIVATIONS IN THE WRIITEN #
    # ASSIGNMENT ARE NOT!                                           #  
    #################################################################
    
    cost = 0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    contextIdxs = [tokens[word] for word in contextWords]
    predicted = np.zeros(inputVectors.shape[1])

    for idx in contextIdxs:
        predicted += inputVectors[idx,:]

    target = tokens[currentWord]
    cost, dv_hat, gradOut = word2vecCostAndGradient(predicted, target, outputVectors, dataset)
    
    for idx in contextIdxs:
        gradIn[idx,:] += dv_hat.reshape(-1)
    ### END YOUR CODE
    
    return cost, gradIn, gradOut

def softmax(x):
    """
    Compute the softmax function for each row of the input x.

    It is crucial that this function is optimized for speed because
    it will be used frequently in later code.
    You might find numpy functions np.exp, np.sum, np.reshape,
    np.max, and numpy broadcasting useful for this task. (numpy
    broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

    You should also make sure that your code works for one
    dimensional inputs (treat the vector as a row), you might find
    it helpful for your later problems.

    You must implement the optimization in problem 1(a) of the 
    written assignment!
    """

    ### YOUR CODE HERE
    if len(x.shape) == 1:
        x = x - np.max(x)
        denom = np.sum(np.exp(x))
        return np.exp(x) / denom
    else:
        x = x - np.max(x, axis=1, keepdims=True)
        denom = np.sum(np.exp(x), axis=1, keepdims=True)
        return np.exp(x) / denom

def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    """
    
    ### YOUR CODE HERE
    x = 1. / (1 + np.exp(-x))
    ### END YOUR CODE
   
    return x

##############################
# Dataset Class for word2vec #
##############################
class Dataset():
    def __init__(self, parser):
        self.parser = parser
        self._sampleTable = None
        self._tablesize = None

    def sampleTable(self, tablesize=10000):
        """
            Table for negative sampling for word2vec.

            :params: tablesize is the size of sampleTable. Should be sufficiently large for sampling.
            :returns: Vx1 sampleTable of word indices
        """
        if self._sampleTable is not None:
            return self._sampleTable

        self._tablesize = tablesize
        parser = self.parser

        freq_table = [0] * len(parser.encode_dict.keys())
        all_sentences = parser.all_sentences
        for sentence in all_sentences:
            for word in sentence:
                freq_table[word] += 1
        freq_table = np.array(freq_table) ** 0.75
        freq_table /= np.sum(freq_table)
        freq_table = np.cumsum(freq_table) * tablesize

        sampleTable = np.array([0] * tablesize)
        
        j = 0
        for i in range(tablesize):
            while i > freq_table[j]:
                j += 1
            sampleTable[i] = j
        
        self._sampleTable = sampleTable
        return sampleTable

    def sampleTokenIdx(self):
        """
            Samples word index from dataset

            :params: None
            :returns: word index (encode_dict)
        """
        return self.sampleTable()[random.randint(0, self._tablesize - 1)]
        
    def getRandomContext(self, C=5):
        parser = self.parser
        all_sentences = parser.all_sentences
        sentID = random.randint(0, len(all_sentences) - 1)
        sent = all_sentences[sentID]
        wordID = random.randint(0, len(sent) - 1)

        context = sent[max(0, wordID - C):wordID]
        if wordID + 1 < len(sent):
            context += sent[wordID+1:min(len(sent), wordID + C + 1)]

        centerword = sent[wordID]
        context = [w for w in context if w != centerword]

        if len(context) > 0:
            return parser.decode_dict[centerword], \
                    [parser.decode_dict[w] for w in context]
        else:
            return self.getRandomContext(C)


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C, word2vecCostAndGradient = softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N//2,:]
    outputVectors = wordVectors[N//2:,:]
    for i in range(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)
        
        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1
        
        c, gin, gout = word2vecModel(centerword, C1, context, tokens, inputVectors, outputVectors, dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N//2, :] += gin / batchsize / denom
        grad[N//2:, :] += gout / batchsize / denom
        
    return cost, grad

def test_word2vec():
    # Interface to the dataset for negative sampling
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], [tokens[random.randint(0,4)] \
           for i in range(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print("==== Gradient check for skip-gram ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)
    print("\n==== Gradient check for CBOW      ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)

    print("\n=== Results ===")
    print(skipgram("c", 3, ["a", "b", "e", "d", "b", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset))
    print(skipgram("c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient))
    print(cbow("a", 2, ["a", "b", "c", "a"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset))
    print(cbow("a", 2, ["a", "b", "a", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient))

if __name__ == "__main__":
#    test_normalize_rows()
#    test_word2vec()
    params = { 'load':False,
            'load_idx':0,
             'dir':'../bstm_models/v4/model',
            'seq_len':25,
            'vocab_size':20002,
            'word_dims':2048}
    parser = None
    if os.path.exists('./.w2v_parser.ckpt'):
        parser = pickle.load(open('./.w2v_parser.ckpt', 'rb'))
    else:
        parser = Parser(["../data/sentence5.csv"], "csv", params['vocab_size'] - 2, params['seq_len']) 
        pickle.dump(parser, open('.w2v_parser.ckpt', 'wb'))
    dataset = Dataset(parser)
    print(dataset.sampleTokenIdx())
    print(dataset.getRandomContext())
