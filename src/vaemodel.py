import tensorflow as tf

"""Abstract class inheriting from the base model class. Provides function definitions for models
   that are variants of or are derived from a variational autoencoder."""
class VAEModel(Model):

    def construct(self, params):
        self.keep_prob = tf.placeholder(tf.float32)
        self.batch_size = tf.placeholder(tf.int32)
        self.kl_alpha = tf.placeholder(tf.float32)
        self.x = tf.placeholder(tf.float32, [None] + params['input_shape'])
        self.y = tf.placeholder(tf.float32, [None] + params['input_shape'])
        self.is_decode = tf.placeholder(tf.bool)
        self.z_input = tf.placeholder(tf.float32, [None, params['latent_dims'])

        with tf.variable_scope('encoder'):
            self.build_encoder(params)

        with tf.variable_scope('latent'):
            sample_normal = tf.random_normal([self.batch_size, params['latent_dims']])
            self.z = (tf.sqrt(tf.exp(self.log_var)) * sample_normal) + self.mu
            self.z_decode = tf.cond(self.is_decode, lambda: self.z_input, lambda: self.z)

        with tf.variable_scope('decoder'):
            self.build_decoder(params)
        
        with tf.variable_scope('loss'):
            with tf.variable_scope('kullback_leibler'):
                kl_div_batch = 1 + self.log_var - tf.square(self.mu) - tf.exp(self.log_var)
                self.kl_div = tf.reduce_mean(-tf.reduce_sum(kl_div_batch, 1))
                tf.summary.scalar('kl_divergence', self.kl_div)
            with tf.variable_scope('reconstruction'):
                self.build_reconstr_loss(params)
                tf.summary.scalar('reconstruction_loss', self.reconstr_loss)
            with tf.variable_scope('optimizer'):
                factor = 0.5*self.kl_alpha
                self.total_loss = factor * self.kl_div + (1 - factor) * self.reconstr_loss
                self.optimizer = tf.train.AdamOptimizer(params['learning_rate'])
                self.train_step = self.optimizer.minimize(self.total_loss)
                tf.summary.scalar('loss', self.total_loss)
                tf.summary.scalar('kl_alpha', self.kl_alpha)

    def build_encoder(self, params):
        pass

    def build_decoder(self, params):
        pass

    def build_reconstr_loss(self, params):
        pass

    def encode(self, x):
        pass

    def decode(self, mu, var):
        pass

    def predict(self, x):
        _, _, z = self.encode(x)
        return self.decode(z)
