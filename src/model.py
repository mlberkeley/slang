import os
import numpy as np
import tensorflow as tf

"""Abstract class from which all models inherit from. Provides common functionality shared
   across all models, including saving, loading, summarizing, and initializing."""
class Model(object):
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

    def construct(self):
        pass

    def lstm_layer(self, init_c, init_h, seq_len, weight_name, x,
                   w_initializer=tf.contrib.layers.xavier_initializer(), 
                   b_initializer=tf.zeros_initializer()):
        x_shape = x.shape.as_list()
        c_shape = init_c.shape.as_list()
        weight_shape = [c_shape[1] + x_shape[2], c_shape[1]]
        wf = tf.get_variable('wf_' + weight_name, weight_shape, initializer=w_initializer)
        bf = tf.get_variable('bf_' + weight_name, c_shape[1], initializer=b_initializer)
        wi = tf.get_variable('wi_' + weight_name, weight_shape, initializer=w_initializer)
        bi = tf.get_variable('bi_' + weight_name, c_shape[1], initializer=b_initializer)
        wc = tf.get_variable('wc_' + weight_name, weight_shape, initializer=w_initializer)
        bc = tf.get_variable('bc_' + weight_name, c_shape[1], initializer=b_initializer)
        wo = tf.get_variable('wo_' + weight_name, weight_shape, initializer=w_initializer)
        bo = tf.get_variable('bo_' + weight_name, c_shape[1], initializer=b_initializer)
        cell = init_c
        hid = init_h
        x_shape = x.shape.as_list()
        xs = tf.split(x, x_shape[1], axis=1)
        out = None
        for i in range(seq_len):
            hx = tf.concat([hid, tf.reshape(xs[i], [self.batch_size, x_shape[2]])], axis=1)
            f = tf.sigmoid(tf.matmul(hx, wf) + bf)
            i = tf.sigmoid(tf.matmul(hx, wi) + bi)
            c = tf.tanh(tf.matmul(hx, wc) + bc)
            o = tf.sigmoid(tf.matmul(hx, wo) + bo)
            cell = tf.multiply(f, cell) + tf.multiply(i, c)
            hid = tf.multiply(o, tf.tanh(cell))
            if out is None:
                out = tf.expand_dims(hid, axis=1)
            else:
                out = tf.concat([out, tf.expand_dims(hid, axis=1)], axis=1)
        return out, cell

    def train(self, batch, step, write_summaries=False):
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
