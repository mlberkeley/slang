import numpy as np
import tensorflow as tf
import model

"""The Variational Sentence Encoding Model. Learns a continuous space sentence encoding using a
seq2seq LSTM variational autoencoder."""
class VSEM(model.Model):
    def construct(self, params):
        self.params = params
        self.keep_prob = tf.placeholder(tf.float32)
        self.batch_size = tf.placeholder(tf.int32)
        self.kl_alpha = tf.placeholder(tf.float32)
        self.x = tf.placeholder(tf.float32, [None, params['seq_len'], params['vocab_size']])
        self.x_p = tf.placeholder(tf.float32, [None, params['seq_len'], params['vocab_size']])
        self.x_l = tf.placeholder(tf.int32, [None, params['seq_len']])
        self.is_decode = tf.placeholder(tf.bool)
        self.z_input = tf.placeholder(tf.float32, [None, params['latent_dims']])

        # build encoder
        with tf.variable_scope('encoder'):
            with tf.variable_scope('lstm'):
                encode_lstm = tf.contrib.rnn.LSTMCell(params['encode_hid'])
                encode_lstm_dropout = tf.contrib.rnn.DropoutWrapper(encode_lstm, self.keep_prob)
                encode_init = encode_lstm.zero_state(self.batch_size, tf.float32)
                encode_outputs, encode_final = tf.nn.dynamic_rnn(encode_lstm, self.x,
                                                                 initial_state = encode_init)
                h = encode_outputs[:, -1, :]
            with tf.variable_scope('mean'):
                self.weights['w_mu'] = tf.get_variable('w_mu', [params['encode_hid'], params['latent_dims']],
                                                       initializer=tf.contrib.layers.xavier_initializer())
                self.weights['b_mu'] = tf.get_variable('b_mu', params['latent_dims'],
                                                       initializer=tf.zeros_initializer())
                self.mu = tf.matmul(h, self.weights['w_mu']) + self.weights['b_mu']
                tf.summary.histogram('mu', self.mu)
            with tf.variable_scope('var'):
                self.weights['w_var'] = tf.get_variable('w_var', [params['encode_hid'], params['latent_dims']],
                                                        initializer=tf.contrib.layers.xavier_initializer())
                self.weights['b_var'] = tf.get_variable('b_var', params['latent_dims'],
                                                        initializer=tf.zeros_initializer())
                self.log_var = tf.matmul(h, self.weights['w_var']) + self.weights['b_var']
                tf.summary.histogram('log_var', self.log_var)
                tf.summary.histogram('std_dev', tf.sqrt(tf.exp(self.log_var)))

        # sample latent variable
        with tf.variable_scope('latent'):
            sample_normal = tf.random_normal([self.batch_size, params['latent_dims']])
            self.z = (tf.sqrt(tf.exp(self.log_var)) * sample_normal) + self.mu
            self.z_decode = tf.cond(self.is_decode, lambda: self.z_input, lambda: self.z)

        # build decoder
        with tf.variable_scope('decoder'):
            with tf.variable_scope('lstm'):
                decode_lstm = tf.contrib.rnn.LSTMCell(params['decode_hid'])
                decode_lstm_dropout = tf.contrib.rnn.DropoutWrapper(decode_lstm, self.keep_prob)
                self.weights['w_z'] = tf.get_variable('w_z', [params['latent_dims'], params['decode_hid']],
                                                      initializer=tf.contrib.layers.xavier_initializer())
                self.weights['b_z'] = tf.get_variable('b_z', params['decode_hid'],
                                                      initializer=tf.zeros_initializer())
                dc = tf.tanh(tf.matmul(self.z_decode, self.weights['w_z']) + self.weights['b_z'])
                decode_init = tf.contrib.rnn.LSTMStateTuple(dc, tf.zeros([self.batch_size, params['decode_hid']]))
                decode_outputs, decode_final = tf.nn.dynamic_rnn(decode_lstm, self.x_p,
                                                                 initial_state = decode_init)
            with tf.variable_scope('pred'):
                self.weights['w_out'] = tf.get_variable('w_out', [params['decode_hid'], params['vocab_size']],
                                                        initializer=tf.contrib.layers.xavier_initializer())
                self.weights['b_out'] = tf.get_variable('b_out', params['vocab_size'],
                                                        initializer=tf.zeros_initializer())
                out_flat = tf.reshape(decode_outputs, [-1, params['decode_hid']])
                y_flat = tf.matmul(out_flat, self.weights['w_out']) + self.weights['b_out']
                self.y_ = tf.reshape(y_flat, [-1, params['seq_len'], params['vocab_size']])
                self.pred = tf.nn.softmax(self.y_)
                tf.summary.scalar('prediction_confidence', tf.reduce_mean(tf.reduce_max(self.pred, 2)))
                reconstr_l = tf.cast(tf.argmax(self.pred, axis=2), tf.int32)
                acc = tf.reduce_mean(tf.cast(tf.equal(self.x_l, reconstr_l), tf.float32))
                tf.summary.scalar('batch_accuracy', acc)

        # calculate loss
        with tf.variable_scope('loss'):
            with tf.variable_scope('kullback_leibler'):
                kl_div_batch = -tf.reduce_sum(1 + self.log_var - tf.square(self.mu) - tf.exp(self.log_var), 1)
                self.kl_div = tf.reduce_mean(kl_div_batch)
                tf.summary.scalar('kl_divergence', self.kl_div)
            with tf.variable_scope('reconstruction'):
                decode_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.x_l, logits=self.y_)
                self.reconstr_loss = tf.reduce_sum(decode_losses)
                tf.summary.scalar('reconstruction_loss', self.reconstr_loss)
            with tf.variable_scope('optimizer'):
                factor = 0.5*self.kl_alpha
                self.total_loss = factor*self.kl_div + (1-factor)*self.reconstr_loss
                self.train_step = tf.train.AdamOptimizer(params['learning_rate']).minimize(self.total_loss)
                tf.summary.scalar('loss', self.total_loss)
                tf.summary.scalar('kl_alpha', self.kl_alpha)

    def to_labels(self, seq_batch, params):
        return seq_batch.nonzero()[2].reshape((params['batch_size'], params['seq_len']))

    def train(self, batch, step, params, write_summaries=False):
        batch_size = batch.shape[0]
        #decode_in = np.concatenate((np.zeros((batch_size, 1, batch.shape[2])), batch[:, :-1, :]), axis=1)
        decode_in = np.zeros(batch.shape)
        feed = { self.keep_prob:params['keep_prob'],
                 self.batch_size:batch_size,
                 self.kl_alpha:min(params['kl_alpha_rate']*step, 1.0),
                 self.x:batch,
                 self.x_p:decode_in,
                 self.x_l:self.to_labels(batch, params),
                 self.is_decode: False,
                 self.z_input: np.zeros((1, params['latent_dims'])) }
        if write_summaries:
            _, summary = self.sess.run([self.train_step, self.merged], feed_dict=feed)
            self.writer.add_summary(summary, step)
        else:
            self.sess.run(self.train_step, feed_dict=feed)

    def encode(self, x):
        x = np.expand_dims(x, axis=0)
        feed = { self.keep_prob:1.0,
                 self.batch_size:1,
                 self.x:x }
        mu, log_var, z = self.sess.run([self.mu, self.log_var, self.z], feed_dict=feed)
        return mu[0,:], log_var[0,:], z[0,:]

    def decode(self, z):
        z = np.expand_dims(z, axis=0)
        _, seq_len, vocab_size = self.x.get_shape().as_list()
        decode_in = np.zeros((1, seq_len, vocab_size))
        dummy_x = np.zeros((1, seq_len, vocab_size))
        feed = { self.keep_prob:1.0,
                 self.batch_size:1,
                 self.x:dummy_x,
                 self.z_input:z,
                 self.x_p:decode_in,
                 self.is_decode:True }
        pred = self.sess.run(self.pred, feed_dict=feed)
        """
        for i in range(seq_len):
            pred = self.sess.run(self.pred, feed_dict=feed)
            if i < seq_len - 1:
                feed[self.x_p][:, i+1, :] = pred[:, i+1, :]
        """
        return pred[0,:,:]


    def predict(self, x):
        _, _, z = self.encode(x)
        return self.decode(z)
