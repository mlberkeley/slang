import os
import numpy as np
import tensorflow as tf

"""Abstract class from which all models inherit from. Provides common functionality shared
   across all models, including saving, loading, summarizing, and initializing."""
class Model(object):
    def __init__(self, params, gpu_fraction=0.3):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.weights = {}

        print('Constructing model')
        self.construct(params)

        self.saver = tf.train.Saver()
        if params['load']:
            dir_index = params['load_idx']
        else:
            dir_index = 0
            while os.path.exists(params['dir'] + '_' + str(dir_index)):
                dir_index += 1
        self.dir = params['dir'] + '_' + str(dir_index)
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        self.ckpt_dir = self.dir + '/checkpoints'

        if params['load']:
            print('Loading saved checkpoint from {}'.format(self.ckpt_dir))
            self.load()
        else:
            print('Initializing new instance of model')
            self.log_dir = self.dir + '/logs'
            self.merged = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
            self.sess.run(tf.global_variables_initializer())

    def construct(self, params):
        pass

    def train(self, batch, step, params, write_summaries=False):
        pass

    def predict(self, x):
        pass

    def load(self):
        if not os.path.exists(self.ckpt_dir):
            raise IOError('The specified checkpoint directory does not exist.')
        latest_ckpt = tf.train.latest_checkpoint(self.ckpt_dir)
        if latest_ckpt:
            print(latest_ckpt)
            self.saver.restore(self.sess, latest_ckpt)
            print('Load success')
        else:
            raise IOError('No checkpoints found in the specified checkpoint directory.')

    def save(self, global_step=None):
        print('Saving checkpoint')
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        self.saver.save(self.sess, self.ckpt_dir, global_step=global_step)
