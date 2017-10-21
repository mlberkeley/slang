import abc
import os
import numpy as np
import tensorflow as tf

class Model(abc.ABC):
    """Abstract class from which all models inherit from. Provides common functionality shared
       across all models, including saving, loading, summarizing, and initializing."""

    def __init__(self, params, gpu_fraction=0.3):
        self.params = params
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.weights = {}

        print('Constructing model')
        self.construct()

        self.saver = tf.train.Saver()
        if self.params['load']:
            dir_index = self.params['load_idx']
        else:
            dir_index = 0
            while os.path.exists(self.params['dir'] + '_' + str(dir_index)):
                dir_index += 1
        self.dir = self.params['dir'] + '_' + str(dir_index)
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        self.ckpt_dir = self.dir + '/checkpoints'

        if self.params['load']:
            print('Loading saved checkpoint from {}'.format(self.ckpt_dir))
            self.load()
        else:
            print('Initializing new instance of model')
            self.log_dir = self.dir + '/logs'
            self.merged = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
            self.sess.run(tf.global_variables_initializer())

    @abc.abstractmethod
    def construct(self):
        """
        Builds the TensorFlow computation graph for the model.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def train(self, batch, step, write_summaries=False):
        """
        Performs a single minibatch training step on the model.

        :param batch: A numpy array containing a single batch of input data
        :param step: An integer representing the iteration number of the training step
        :param write_summaries: A boolean toggle for writing Tensorboard summaries for the current
                                training step
        """
        raise NotImplementedError

    def load(self):
        """
        Loads the parameters of the latest saved checkpoint for the model.
        """
        if not os.path.exists(self.ckpt_dir):
            raise IOError('The specified checkpoint directory does not exist.')
        latest_ckpt = tf.train.latest_checkpoint(self.dir)
        if latest_ckpt:
            print(latest_ckpt)
            self.saver.restore(self.sess, latest_ckpt)
            print('Load success')
        else:
            raise IOError('No checkpoints found in the specified checkpoint directory.')

    def save(self, global_step=None):
        """
        Save the current values of the parameters into a checkpoint file.

        :param global_step: The integer representing the step value with which to save the
                            checkpoint file as
        """
        print('Saving checkpoint')
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        self.saver.save(self.sess, self.ckpt_dir, global_step=global_step)
