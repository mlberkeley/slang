import numpy as np
import tensorflow as tf
import model

class VSEM(model.Model):
    """The Variational Sentence Encoding Model. Learns a continuous space sentence encoding using 
    a seq2seq LSTM variational autoencoder."""

    def construct(self):
        self.keep_prob = tf.placeholder(tf.float32)
        self.batch_size = tf.placeholder(tf.int32)
        self.kl_alpha = tf.placeholder(tf.float32)
        self.x = tf.placeholder(tf.float32, [None, self.params['seq_len'],
                                             self.params['wordvec_dims']])
        self.is_decode = tf.placeholder(tf.bool)
        self.z_input = tf.placeholder(tf.float32, [None, self.params['latent_dims']])
        
        xavier = tf.contrib.layers.xavier_initializer()

        # build encoder
        with tf.variable_scope('encoder'):
            with tf.variable_scope('rnn'):
                encode_rnn = tf.contrib.rnn.LSTMCell(self.params['encode_hid'])
                encode_rnn_dropout = tf.contrib.rnn.DropoutWrapper(encode_rnn, self.keep_prob)
                encode_init = tf.contrib.rnn.LSTMStateTuple(
                                  tf.zeros([self.batch_size, self.params['encode_hid']),
                                  tf.zeros([self.batch_size, self.params['encode_hid']))
                encode_outputs, encode_final = tf.nn.dynamic_rnn(encode_rnn, self.x,
                                                                 initial_state = encode_init)
                h = encode_outputs[:, -1, :]
            with tf.variable_scope('mean'):
                w_mu_shape = [self.params['encode_hid'], self.params['latent_dims']]
                self.weights['w_mu'] = tf.get_variable('w_mu', w_mu_shape, initializer=xavier)
                self.weights['b_mu'] = tf.get_variable('b_mu', self.params['latent_dims'],
                                                       initializer=tf.zeros_initializer())
                self.mu = tf.matmul(h, self.weights['w_mu']) + self.weights['b_mu']
                tf.summary.histogram('mu', self.mu)
            with tf.variable_scope('var'):
                w_var_shape = [self.params['encode_hid'], self.params['latent_dims']]
                self.weights['w_var'] = tf.get_variable('w_var', w_var_shape, initializer=xavier)
                self.weights['b_var'] = tf.get_variable('b_var', self.params['latent_dims'],
                                                        initializer=tf.zeros_initializer())
                self.log_var = tf.matmul(h, self.weights['w_var']) + self.weights['b_var']
                tf.summary.histogram('log_var', self.log_var)
                tf.summary.histogram('std_dev', tf.sqrt(tf.exp(self.log_var)))

        # sample latent variable
        with tf.variable_scope('latent'):
            sample_normal = tf.random_normal([self.batch_size, self.params['latent_dims']])
            self.z = (tf.sqrt(tf.exp(self.log_var)) * sample_normal) + self.mu
            self.z_decode = tf.cond(self.is_decode, lambda: self.z_input, lambda: self.z)

        # build decoder
        with tf.variable_scope('decoder'):
            with tf.variable_scope('rnn'):
                decode_rnn = tf.contrib.rnn.LSTMCell(self.params['decode_hid'])
                decode_rnn_dropout = tf.contrib.rnn.DropoutWrapper(decode_rnn, self.keep_prob)
                w_z_shape = [self.params['latent_dims'], self.params['decode_hid']]
                self.weights['w_z'] = tf.get_variable('w_z', w_z_shape, initializer=xavier)
                self.weights['b_z'] = tf.get_variable('b_z', self.params['decode_hid'],
                                                      initializer=tf.zeros_initializer())
                dc = tf.tanh(tf.matmul(self.z_decode, self.weights['w_z']) + self.weights['b_z'])
                empty = tf.zeros([self.batch_size, self.params['seq_len'],
                                                   self.params['decode_hid']])
                decode_outputs, decode_final = tf.nn.dynamic_rnn(decode_rnn, empty,
                                                                 initial_state=dc)
            with tf.variable_scope('pred'):
                w_out_shape = [self.params['decode_hid'], self.params['wordvec_dims']]
                self.weights['w_out'] = tf.get_variable('w_out', w_out_shape, initializer=xavier)
                self.weights['b_out'] = tf.get_variable('b_out', self.params['wordvec_dims'],
                                                        initializer=tf.zeros_initializer())
                out_flat = tf.reshape(decode_outputs, [-1, self.params['decode_hid']])
                y_flat = tf.matmul(out_flat, self.weights['w_out']) + self.weights['b_out']
                self.y_ = tf.reshape(y_flat, [-1, self.params['seq_len'],
                                              self.params['wordvec_dims']])

        # calculate loss
        with tf.variable_scope('loss'):
            with tf.variable_scope('kullback_leibler'):
                kl_div_batch = 1 + self.log_var - tf.square(self.mu) - tf.exp(self.log_var)
                self.kl_div = tf.reduce_mean(-tf.reduce_mean(kl_div_batch, 1))
                tf.summary.scalar('kl_divergence', self.kl_div)
            with tf.variable_scope('reconstruction'):
                xy_dot = tf.reduce_sum(tf.multiply(self.x, self.y_), axis=2)
                norms = tf.multiply(tf.norm(self.x, axis=2), tf.norm(self.y_, axis=2))
                cosine_loss = 1 - tf.divide(xy_dot, norms)
                self.reconstr_loss = tf.reduce_mean(tf.reduce_sum(cosine_loss, axis=1))
                tf.summary.scalar('reconstruction_loss', self.reconstr_loss)
            with tf.variable_scope('optimizer'):
                factor = 0.5*self.kl_alpha
                self.total_loss = factor*self.kl_div + (1-factor)*self.reconstr_loss
                self.optimizer = tf.train.AdamOptimizer(self.params['learning_rate'])
                self.train_step = self.optimizer.minimize(self.total_loss)
                tf.summary.scalar('loss', self.total_loss)
                tf.summary.scalar('kl_alpha', self.kl_alpha)

    def train(self, batch, step, write_summaries=False):
        batch_size = batch.shape[0]
        feed = { self.keep_prob:self.params['keep_prob'],
                 self.batch_size:batch_size,
                 self.kl_alpha:min(self.params['kl_alpha_rate']*step, 1.0),
                 self.x:batch,
                 self.is_decode: False,
                 self.z_input: np.zeros((1, self.params['latent_dims'])) }
        if write_summaries:
            _, summary = self.sess.run([self.train_step, self.merged], feed_dict=feed)
            self.writer.add_summary(summary, step)
        else:
            self.sess.run(self.train_step, feed_dict=feed)

    def encode_batch(self, x):
        feed = { self.keep_prob:1.0,
                 self.batch_size:1,
                 self.x:x }
        return self.sess.run([self.mu, self.log_var, self.z], feed_dict=feed)

    def decode_batch(self, z):
        batch_size, seq_len, dims = self.x.get_shape().as_list()
        dummy_x = np.zeros(self.x.get_shape())
        feed = { self.keep_prob:1.0,
                 self.batch_size:batch_size,
                 self.x:dummy_x,
                 self.z_input:z,
                 self.is_decode:True }
        pred = self.sess.run(self.y_, feed_dict=feed)
        return pred

    def encode(self, x):
        x = np.expand_dims(x, axis=0)
        mu, lv, z = self.encode_batch(x)
        return mu[0,:], lv[0,:], z[0,:]

    def decode(self, z):
        z = np.expand_dims(z, axis=0)
        pred = self.decode_batch(z)
        return pred[0,:,:]

    def predict(self, x):
        x = np.expand_dims(x, axis=0)
        _, _, z = self.encode(x)
        return self.decode(z[0,:])
