import numpy as np
import tensorflow as tf
import model

"""The Bijective Skip-Thought Model. Builds off of skip-thoughts by allowing for both encoding and
decoding of the senteces into thought vectors."""
class BSTM(model.Model):
    def build_encoder(self, x, keep_prob, name):
        x_onehot = tf.one_hot(x, self.params['vocab_size'], axis=-1)
        with tf.variable_scope('word_embed'):
            w_emb_shape = [self.params['vocab_size'], self.params['word_dims']]
            self.weights['w_emb_'+name] = tf.get_variable('w_emb_'+name, w_emb_shape,
                                                          initializer=self.xavier)
            self.weights['b_emb_'+name] = tf.get_variable('b_emb_'+name, self.params['word_dims'],
                                                          initializer=tf.zeros_initializer())
            x_flat = tf.reshape(x_onehot, [-1, self.params['vocab_size']])
            emb_flat = tf.matmul(x_flat, self.weights['w_emb_'+name]) + \
                       self.weights['b_emb_'+name]
            emb = tf.reshape(emb_flat, [-1, self.params['seq_len'],
                                        self.params['word_dims']])
        with tf.variable_scope('lstm'):
            encode_lstm = tf.contrib.rnn.LSTMCell(self.params['encode_hid'])
            encode_init = encode_lstm.zero_state(self.batch_size, tf.float32)
            encode_outputs, encode_final = tf.nn.dynamic_rnn(encode_lstm, emb,
                                                             initial_state=encode_init)
            h = tf.nn.dropout(encode_outputs[:, -1, :], keep_prob)
        return h

    def build_decoder(self, z, keep_prob, name):
        with tf.variable_scope('lstm'):
            decode_lstm = tf.contrib.rnn.LSTMCell(self.params['latent_dims'])
            decode_init_c = tf.zeros([self.batch_size, self.params['latent_dims']])
            decode_init_h = z
            decode_init = tf.contrib.rnn.LSTMStateTuple(decode_init_c, decode_init_h)
            empty = tf.zeros([self.batch_size, self.params['seq_len'], 1])
            decode_outputs, decode_final = tf.nn.dynamic_rnn(decode_lstm, empty,
                                                             initial_state=decode_init)
            h = tf.nn.dropout(tf.reshape(decode_outputs, [-1, self.params['latent_dims']]),
                              keep_prob)
        with tf.variable_scope('pred'):
            w_decode_shape = [self.params['latent_dims'], self.params['vocab_size']]
            self.weights['w_pred_'+name] = tf.get_variable('w_pred_'+name, w_decode_shape,
                                                           initializer=self.xavier)
            self.weights['b_pred_'+name] = tf.get_variable('b_pred_'+name,
                                                           self.params['vocab_size'],
                                                           initializer=tf.zeros_initializer())
            y_flat = tf.matmul(h, self.weights['w_pred_'+name]) + self.weights['b_pred_'+name]
            y_ = tf.reshape(y_flat, [-1, self.params['seq_len'],
                                     self.params['vocab_size']])
            pred = tf.nn.softmax(y_)

        return y_, pred

    def construct(self):
        self.keep_prob = tf.placeholder(tf.float32)
        self.batch_size = tf.placeholder(tf.int32)
        self.kl_alpha = tf.placeholder(tf.float32)
        self.x = tf.placeholder(tf.int32, [None, self.params['seq_len']])
        self.pre = tf.placeholder(tf.int32, [None, self.params['seq_len']])
        self.post = tf.placeholder(tf.int32, [None, self.params['seq_len']])
        self.is_decode = tf.placeholder(tf.bool)
        self.z_input = tf.placeholder(tf.float32, [None, self.params['latent_dims']])

        self.xavier = tf.contrib.layers.xavier_initializer()

        # build encoder
        with tf.variable_scope('context_encoder'):
            with tf.variable_scope('pre'):
                pre_vec = self.build_encoder(self.pre, self.keep_prob, 'pre')
            with tf.variable_scope('post'):
                post_vec = self.build_encoder(self.post, self.keep_prob, 'post')
            ctxt_shape = [2 * self.params['encode_hid'], self.params['latent_dims']]
            ctxt_vec = tf.concat([pre_vec, post_vec], 1)
            with tf.variable_scope('mu'):
                self.weights['w_ctxt_mu'] = tf.get_variable('w_ctxt_mu', ctxt_shape,
                                                             initializer=self.xavier)
                self.weights['b_ctxt_mu'] = tf.get_variable('b_ctxt_mu', self.params['latent_dims'],
                                                             initializer=tf.zeros_initializer())
                self.ctxt_mu = tf.matmul(ctxt_vec, self.weights['w_ctxt_mu']) + \
                               self.weights['b_ctxt_mu']
                tf.summary.histogram('ctxt_mu', self.ctxt_mu)
            with tf.variable_scope('var'):
                self.weights['w_ctxt_var'] = tf.get_variable('w_ctxt_var', ctxt_shape,
                                                             initializer=self.xavier)
                self.weights['b_ctxt_var'] = tf.get_variable('b_ctxt_var', self.params['latent_dims'],
                                                             initializer=tf.zeros_initializer())
                self.ctxt_logvar = tf.matmul(ctxt_vec, self.weights['w_ctxt_var']) + \
                                   self.weights['b_ctxt_var']
                self.ctxt_stddev = tf.sqrt(tf.exp(self.ctxt_logvar))
                tf.summary.histogram('ctxt_stddev', self.ctxt_stddev)
        with tf.variable_scope('current_encoder'):
            with tf.variable_scope('curr'):
                curr_vec = self.build_encoder(self.x, self.keep_prob, 'curr')
            curr_shape = [self.params['encode_hid'], self.params['latent_dims']]
            with tf.variable_scope('mu'):
                self.weights['w_curr_mu'] = tf.get_variable('w_curr_mu', curr_shape,
                                                             initializer=self.xavier)
                self.weights['b_curr_mu'] = tf.get_variable('b_curr_mu', self.params['latent_dims'],
                                                             initializer=tf.zeros_initializer())
                self.curr_mu = tf.matmul(curr_vec, self.weights['w_curr_mu']) + \
                               self.weights['b_curr_mu']
                tf.summary.histogram('curr_mu', self.curr_mu)
            with tf.variable_scope('var'):
                self.weights['w_curr_var'] = tf.get_variable('w_curr_var', curr_shape,
                                                             initializer=self.xavier)
                self.weights['b_curr_var'] = tf.get_variable('b_curr_var', self.params['latent_dims'],
                                                             initializer=tf.zeros_initializer())
                self.curr_logvar = tf.matmul(curr_vec, self.weights['w_curr_var']) + \
                                   self.weights['b_curr_var']
                self.curr_stddev = tf.sqrt(tf.exp(self.curr_logvar))
                tf.summary.histogram('curr_stddev', self.curr_stddev)

        with tf.variable_scope('latent'):
            smpl_ctxt = tf.random_normal([self.batch_size, self.params['latent_dims']])
            smpl_curr = tf.random_normal([self.batch_size, self.params['latent_dims']])
            self.z_ctxt = (self.ctxt_stddev * smpl_ctxt) + self.ctxt_mu
            self.z_curr = (self.curr_stddev * smpl_ctxt) + self.curr_mu
            self.z_decode = tf.cond(self.is_decode, lambda: self.z_input, lambda: self.z_ctxt)
        
        # build decoder
        with tf.variable_scope('context_decoder'):
            with tf.variable_scope('pre'):
                self.y_pre, self.pred_pre = self.build_decoder(self.z_curr, self.keep_prob,
                                                               'pre')
                pred_pre_conf = tf.reduce_mean(tf.reduce_max(self.pred_pre, 2))
                tf.summary.scalar('prediction_confidence_pre', pred_pre_conf)
                reconstr_pre = tf.cast(tf.argmax(self.pred_pre, axis=2), tf.int32)
                acc_pre = tf.reduce_mean(tf.cast(tf.equal(self.pre, reconstr_pre), tf.float32))
                tf.summary.scalar('accuracy_pre', acc_pre)
            with tf.variable_scope('post'):
                self.y_post, self.pred_post = self.build_decoder(self.z_curr, self.keep_prob,
                                                                 'post')
                pred_post_conf = tf.reduce_mean(tf.reduce_max(self.pred_post, 2))
                tf.summary.scalar('prediction_confidence_post', pred_post_conf)
                reconstr_post = tf.cast(tf.argmax(self.pred_post, axis=2), tf.int32)
                acc_post = tf.reduce_mean(tf.cast(tf.equal(self.post, reconstr_post), tf.float32))
                tf.summary.scalar('accuracy_post', acc_post)
        with tf.variable_scope('current_decoder'):
            with tf.variable_scope('curr'):
                self.y_curr, self.pred_curr = self.build_decoder(self.z_decode, self.keep_prob,
                                                                 'curr')
                pred_curr_conf = tf.reduce_mean(tf.reduce_max(self.pred_curr, 2))
                tf.summary.scalar('prediction_confidence_curr', pred_curr_conf)
                reconstr_curr = tf.cast(tf.argmax(self.pred_curr, axis=2), tf.int32)
                acc_curr = tf.reduce_mean(tf.cast(tf.equal(self.x, reconstr_curr), tf.float32))
                tf.summary.scalar('accuracy_curr', acc_curr)

        #calculate loss
        with tf.variable_scope('loss'):
            with tf.variable_scope('reconstruction'):
                pre_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.pre,
                                                                            logits=self.y_pre)
                self.pre_l = tf.reduce_sum(pre_losses)
                tf.summary.scalar('loss_pre', self.pre_l)
                curr_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.x,
                                                                             logits=self.y_curr)
                self.curr_l = tf.reduce_sum(curr_losses)
                tf.summary.scalar('loss_curr', self.curr_l)
                post_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.post,
                                                                             logits=self.y_post)
                self.post_l = tf.reduce_sum(post_losses)
                tf.summary.scalar('loss_post', self.post_l)
                self.skipthought_l = self.pre_l + self.post_l
                self.reconstr_l = self.curr_l
            with tf.variable_scope('latent'):
                kl_div_ctxt = 1 + self.ctxt_logvar - tf.square(self.ctxt_mu) - \
                              tf.exp(self.ctxt_logvar)
                kl_div_curr = 1 + self.curr_logvar - tf.square(self.curr_mu) - \
                              tf.exp(self.curr_logvar)
                self.kl_div_ctxt_l = tf.reduce_mean(-tf.reduce_sum(kl_div_ctxt, 1))
                tf.summary.scalar('loss_kl_context', self.kl_div_ctxt_l)
                self.kl_div_curr_l = tf.reduce_mean(-tf.reduce_sum(kl_div_curr, 1))
                tf.summary.scalar('loss_kl_current', self.kl_div_curr_l)
                self.latent_diff_l = tf.square(tf.norm(self.ctxt_mu - self.curr_mu)) + \
                                     tf.square(tf.norm(self.ctxt_stddev - self.curr_stddev))
                tf.summary.scalar('loss_latent', self.latent_diff_l)
            with tf.variable_scope('optimizer'):
                factor = 0.5 * self.kl_alpha
                self.encoding_l = self.kl_div_ctxt_l + self.kl_div_curr_l + self.latent_diff_l
                self.total_l = (1 - factor) * (self.skipthought_l + self.reconstr_l) + \
                               factor * self.encoding_l
                self.optimizer = tf.train.AdamOptimizer(self.params['learning_rate'])
                self.train_step = self.optimizer.minimize(self.total_l)
                tf.summary.scalar('loss', self.total_l)

    def train(self, batch, step, write_summaries=False):
        batch_size = batch.shape[0]
        pre, x, post = np.split(batch, 3, axis=1)
        feed = { self.keep_prob:self.params['keep_prob'],
                 self.batch_size:batch_size,
                 self.kl_alpha:min(self.params['kl_alpha_rate']*step, 1.0),
                 self.pre:np.squeeze(pre),
                 self.x:np.squeeze(x),
                 self.post:np.squeeze(post),
                 self.is_decode:False,
                 self.z_input:np.zeros((1, self.params['latent_dims'])) }
        if write_summaries:
            _, summary = self.sess.run([self.train_step, self.merged],
                                          feed_dict=feed)
            self.writer.add_summary(summary, step)
        else:
            self.sess.run(self.train_step, feed_dict=feed)

    def encode(self, x):
        x = np.expand_dims(x, axis=0)
        feed = {self.keep_prob:1.0,
                self.batch_size:1,
                self.x:x }
        mu, stddev, z = self.sess.run([self.curr_mu, self.curr_stddev, self.curr_z],
                                      feed_dict=feed)
        return mu[0,:], stddev[0,:], z[0,:]

    def decode(self, z):
        z = np.expand_dims(z, axis=0)
        _, seq_len = self.x.get_shape().as_list()
        dummy_x = np.zeros((1, seq_len))
        feed = { self.keep_prob:1.0,
                 self.batch_size:1,
                 self.x:dummy_x,
                 self.z_input:z,
                 self.is_decode:True }
        pred = self.sess.run(self.pred, feed_dict=feed)
        return pred[0,:,:]

    def predict(self, x):
        _, _, z = self.encode(x)
        return self.decode(z)
