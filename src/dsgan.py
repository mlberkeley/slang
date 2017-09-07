import model

"""The Distribution Sequence Generative Adversarial Network. Produces an adversarially trained
sequence of sentence distributions, to be decoded by the VSEM."""
class DSGAN(model.Model):

    def build_generator(self):
        sample = tf.random_normal([self.batch_size, self.params['num_sent'], self.params['smpl_dims']])
        with tf.variable_scope('lstm'):
            gen_lstm = tf.contrib.rnn.LSTMCell(self.params['gen_hid'])
            gen_lstm_dropout = tf.contrib.rnn.DropoutWrapper(gen_lstm, self.gen_keep_prob)
            gen_init = gen_lstm.zero_state(self.batch_size, tf.float32)
            gen_outputs, gen_final = tf.nn.dynamic_rnn(gen_lstm, sample, initial_state=gen_init)
            gen_out_flat = tf.reshape(gen_outputs, [-1, self.params['gen_hid']])
        w_out_shape = [self.params['gen_hid'], self.params['latent_dims']]

        with tf.variable_scope('mean'):
            self.weights['w_mu'] = tf.get_variable('w_mu', w_out_shape, initializer=self.xavier)
            self.weights['b_mu'] = tf.get_variable('b_mu', self.params['latent_dims'],
                                                   initializer=tf.zeros_initializer())
            mu_flat = tf.matmul(gen_out_flat, self.weights['w_mu']) + self.weights['b_mu']
            mu = tf.reshape(mu_flat, [-1, self.params['num_sent'], self.params['latent_dims']])

        with tf.variable_scope('var'):
            self.weights['w_lv'] = tf.get_variable('w_lv', w_out_shape, initializer=self.xavier)
            self.weights['b_lv'] = tf.get_variable('b_lv', self.params['latent_dims'],
                                                   initializer=tf.zeros_initializer())
            lv_flat = tf.matmul(gen_out_flat, self.weights['w_lv']) + self.weights['b_lv']
            lv = tf.reshape(lv_flat, [-1, self.params['num_sent'], self.params['latent_dims']])

        return mu, lv

    def build_discriminator(self, data):
        with tf.variable_scope('lstm'):
            dis_lstm = tf.contrib.rnn.LSTMCell(self.params['dis_hid'])
            dis_lstm_dropout = tf.contrib.rnn.DropoutWrapper(dis_lstm, self.dis_keep_prob)
            dis_init = dis_lstm.zero_state(self.batch_size, tf.float32)
            dis_outputs, dis_final = tf.nn.dynamic_rnn(dis_lstm, data, initial_state=dis_init)
            h = dis_outputs[:, -1, :]
        with tf.variable_scope('pred'):
            w_dis_shape = [self.params['dis_hid'], 1]
            self.weights['w_dis'] = tf.get_variable('w_dis', w_dis_shape, initializer=self.xavier)
            self.weights['b_dis'] = tf.get_variable('b_dis', 1, initializer=tf.zeros_initializer())
            return tf.sigmoid(tf.matmul(h, self.weights['w_dis']) + self.weights['b_dis'])

    def construct(self):
        self.gen_keep_prob = tf.placeholder(tf.float32)
        self.dis_keep_prob = tf.placeholder(tf.float32)
        self.batch_size = tf.placeholder(tf.int32)
        self.x_mus = tf.placeholder(tf.float32, [None, self.params['num_sent'], self.params['latent_dims']])
        self.x_lvs = tf.placeholder(tf.float32, [None, self.params['num_sent'], self.params['latent_dims']])
        self.xavier = tf.contrib.layers.xavier_initializer()

        # build generator and discriminator
        with tf.variable_scope('generator'):
            self.y_mus, self.y_lvs = build_generator()
        self.dis_x = tf.concat([self.x_mus, self.x_lvs], axis=2)
        self.dis_y = tf.concat([self.y_mus, self.y_lvs], axis=2)
        with tf.variable_scope('discriminator'):
            self.pred_x = build_discriminator(self.dis_x)
            self.pred_y = build_discriminator(self.dis_y)

        # calculate loss
        with tf.variable_scope('loss'):
            # gradient penalty
            with tf.variable_scope('grad_penalty'):
                self.alpha = tf.random_uniform([self.batch_size, 1])
                self.cvx_comb = (1-self.alpha)*self.dis_x + self.alpha*self.dis_y
                self.grads = tf.gradients(build_discriminator(self.cvx_comb), [cvx_comb])[0]
                self.l2 = tf.sqrt(tf.reduce_sum(tf.square(grads), reduction_indices=[1]))
                self.grad_penalty = tf.reduce_mean((self.l2-1.0)**2)
            
            with tf.variable_scope('gen_loss'):
                self.gen_loss = -tf.reduce_mean(self.pred_y)
                self.gen_optimizer = tf.train.AdamOptimizer(self.params['learning_rate'])
                self.gen_train = self.gen_optimizer.minimize(self.gen_loss)
            with tf.variable_scope('dis_loss'):
                self.dis_loss = tf.reduce_mean(self.pred_y) - tf.reduce_mean(self.pred_x) +
                                self.params['lambda']*self.grad_penalty
                self.dis_optimizer = tf.train.AdamOptimizer(self.params['learning_rate'])
                self.dis_train = self.dis_optimizer.minimize(self.dis_loss)

    def train_generator(self, batch, step, write_summaries=False):
        batch_size = batch.shape[1]
        feed = {self.gen_keep_prob:self.params['keep_prob'],
                self.dis_keep_prob:1.0,
                self.batch_size:batch_size,
                self.x_mus:batch[0],
                self.x_lvs:batch[1] }
        if write_summaries:
            _, summary = self.sess.run([self.gen_train, self.merged], feed_dict=feed)
            self.writer.add_summary(summary, step)
        else:
            self.sess.run(self.gen_train, feed_dict=feed)

    def train_discriminator(self, batch):
        batch_size = batch.shape[1]
        feed = {self.gen_keep_prob:1.0,
                self.dis_keep_prob:self.params['keep_prob'],
                self.batch_size:batch_size,
                self.x_mus:batch[0],
                self.x_lvs:batch[1] }
        self.sess.run(self.gen_train, feed_dict=feed)

    def generate(self):
        feed = {self.gen_keep_prob:1.0,
                self.batch_size:1 }
        return self.sess.run([self.y_mus, self.y_lvs], feed_dict=feed)
