import tensorflow as tf
import model

class DSGAN(model.Model):
    """The Distribution Sequence Generative Adversarial Network. Produces an adversarially trained
    sequence of sentence distributions, to be decoded by the VSEM."""

    def conv1d(self, x, filter_size, stride, feature_map_dim, name):
        with tf.variable_scope(name):
            filter_dims = [filter_size, x.get_shape()[-1], feature_map_dim]
            self.weights['w_'+name] = tf.get_variable('w_'+name, filter_dims,
                                                      initializer=self.xavier)
            self.weights['b_'+name] = tf.get_variable('b_'+name, feature_map_dim,
                                                      initializer=tf.zeros_initializer())
            conv = tf.nn.conv1d(x, self.weights['w_'+name], stride, padding='SAME')
            return tf.nn.relu(conv + self.weights['b_'+name])

    def build_generator(self):
        sample = tf.random_normal([self.batch_size, self.params['num_sent'], self.params['smpl_dims']])
        with tf.variable_scope('rnn'):
            gen_lstm = tf.contrib.rnn.LSTMCell(self.params['gen_hid'])
            gen_lstm_dropout = tf.contrib.rnn.DropoutWrapper(gen_lstm, self.gen_keep_prob)
            gen_cell_zeros = tf.zeros([self.batch_size, self.params['gen_hid']])
            gen_hidd_zeros = tf.zeros([self.batch_size, self.params['gen_hid']])
            gen_init = tf.contrib.rnn.LSTMStateTuple(gen_cell_zeros, gen_hidd_zeros)
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
        with tf.variable_scope('latent'):
            sample_normal = tf.random_normal([self.batch_size, self.params['num_sent'],
                                              self.params['latent_dims']])
            z = (tf.sqrt(tf.exp(lv)) * sample_normal) + mu

        return mu, lv, z

    def build_discriminator(self, data):
        with tf.variable_scope('cnn'):
            x = tf.transpose(data, perm=[0, 2, 1])
            conv1 = self.conv1d(data, 10, 1, 16, 'dis_1')
            conv2 = self.conv1d(conv1, 10, 1, 32, 'dis_2')
            pool3 = tf.nn.pool(conv2, [4], 'MAX', 'SAME')
            conv4 = self.conv1d(pool3, 10, 1, 64, 'dis_4')
            conv5 = self.conv1d(conv4, 10, 1, 64, 'dis_5')
            pool6 = tf.nn.pool(conv5, [4], 'MAX', 'SAME')
            conv7 = self.conv1d(pool6, 10, 1, 32, 'dis_6')
            conv8 = self.conv1d(conv7, 10, 1, 16, 'dis_7')
            pool9 = tf.nn.pool(conv8, [4], 'MAX', 'SAME')
            conv_shape = pool3.get_shape().as_list()
            cnn_out = tf.reshape(pool9, [-1, conv_shape[1]*conv_shape[2]])
        """
        with tf.variable_scope('rnn'):
            dis_lstm = tf.contrib.rnn.LSTMCell(self.params['dis_hid'])
            dis_lstm_dropout = tf.contrib.rnn.DropoutWrapper(dis_lstm, self.dis_keep_prob)
            dis_cell_zeros = tf.zeros([self.batch_size, self.params['dis_hid']])
            dis_hidd_zeros = tf.zeros([self.batch_size, self.params['dis_hid']])
            dis_init = tf.contrib.rnn.LSTMStateTuple(dis_cell_zeros, dis_hidd_zeros)
            dis_outputs, dis_final = tf.nn.dynamic_rnn(dis_lstm, data, initial_state=dis_init)
            h = dis_outputs[:, -1, :]
        """
        with tf.variable_scope('pred'):
            w_dis_shape = [cnn_out.get_shape()[1], 1]
            self.weights['w_dis'] = tf.get_variable('w_dis', w_dis_shape, initializer=self.xavier)
            self.weights['b_dis'] = tf.get_variable('b_dis', 1, initializer=tf.zeros_initializer())
            return tf.sigmoid(tf.matmul(cnn_out, self.weights['w_dis']) + self.weights['b_dis'])

    def construct(self):
        self.gen_keep_prob = tf.placeholder(tf.float32)
        self.dis_keep_prob = tf.placeholder(tf.float32)
        self.batch_size = tf.placeholder(tf.int32)
        self.x_mus = tf.placeholder(tf.float32, [None, self.params['num_sent'], self.params['latent_dims']])
        self.x_lvs = tf.placeholder(tf.float32, [None, self.params['num_sent'], self.params['latent_dims']])
        self.xavier = tf.contrib.layers.xavier_initializer()

        # build generator and discriminator
        with tf.variable_scope('generator'):
            self.y_mus, self.y_lvs, self.zs = self.build_generator()
        self.dis_x = tf.concat([self.x_mus, self.x_lvs], axis=2)
        self.dis_y = tf.concat([self.y_mus, self.y_lvs], axis=2)
        with tf.variable_scope('discriminator') as scope:
            with tf.variable_scope('combination'):
                self.alpha = tf.random_uniform([self.batch_size, 1, 1])
                self.cvx_comb = (1-self.alpha)*self.dis_x + self.alpha*self.dis_y
            self.pred_x = self.build_discriminator(self.dis_x)
            scope.reuse_variables()
            self.pred_y = self.build_discriminator(self.dis_y)
            scope.reuse_variables()
            self.comb_out = self.build_discriminator(self.cvx_comb)

        # calculate loss
        with tf.variable_scope('loss'):
            # gradient penalty
            with tf.variable_scope('grad_penalty'):
                self.grads = tf.gradients(self.comb_out, [self.cvx_comb])[0]
                self.l2 = tf.sqrt(tf.reduce_sum(tf.square(self.grads), reduction_indices=[1]))
                tf.summary.scalar('gradient_l2_norm', tf.reduce_mean(self.l2))
                self.grad_penalty = tf.reduce_mean((self.l2-1.0)**2)
            
            with tf.variable_scope('gen_loss'):
                self.gen_loss = -tf.reduce_mean(self.pred_y)
                tf.summary.scalar('loss_generator', self.gen_loss)
                self.gen_optimizer = tf.train.AdamOptimizer(self.params['learning_rate'])
                gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                             scope='generator')
                self.gen_train = self.gen_optimizer.minimize(self.gen_loss, var_list=gen_vars)
            with tf.variable_scope('dis_loss'):
                self.dis_loss = tf.reduce_mean(self.pred_y) - tf.reduce_mean(self.pred_x) + \
                                self.params['lambda']*self.grad_penalty
                tf.summary.scalar('loss_discriminator', self.dis_loss)
                self.dis_optimizer = tf.train.AdamOptimizer(self.params['learning_rate'])
                dis_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                             scope='discriminator')
                self.dis_train = self.dis_optimizer.minimize(self.dis_loss, var_list=dis_vars)

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

    def train(self, batch, step, write_summaries=False):
        self.train_discriminator(batch)
        self.train_generator(batch, step, write_summaries=write_summaries)

    def generate(self):
        feed = {self.gen_keep_prob:1.0,
                self.batch_size:1 }
        return self.sess.run(self.zs, feed_dict=feed)[0,:,:]
