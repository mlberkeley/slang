import os
import tensorflow as tf

"""Abstract class from which all models inherit from. Provides common functionality shared 
   across all models, including saving, loading, summarizing, and initializing."""
class Model(object):
    def __init__(self, sess, params):
        self.sess = sess

        print('Constructing model')
        self.construct(params)

        self.saver = tf.train.Saver()
        self.ckpt_dir = params['ckpt_dir']
        self.writer = tf.train.FileWriter(params['log_dir'])

        if (params['load']):
            print('Loading saved checkpoint from {}'.format(params['ckpt_dir']))
            self.load(params)
        else
            print('Initializing new instance of model')
            self.initialize(params)

    @abstractmethod
    def construct(self, params):
        pass

    def load(self, params):
        if not os.path.exists(self.ckpt_dir):
            raise IOError('The specified checkpoint directory does not exist.')
        latest_ckpt = tf.train.latest_checkpoint(params['ckpt_dir'])
        if latest_ckpt:
            self.saver.restore(self.sess, latest_ckpt)
            print('Load success')
        else:
            raise IOError('No checkpoints found in the specified checkpoint directory.')

    def initialize(self, params):
        self.merged = tf.summary.merge_all()
        self.writer = tf.train.FileWriter(params['log_dir'])
        self.sess.run(tf.global_variables_initializer())

    def save(self, global_step=None):
        print('Saving checkpoint')
        if not os.path.exists(self.ckpt_dir):
            os.makdirs(self.ckpt_dir)
        self.saver.save(self.sess, self.ckpt_dir, global_step=global_step)

