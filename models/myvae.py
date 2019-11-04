import coders.vae_coding
import tensorflow as tf
import numpy as np
class myvae():
    def __init__(self, latent_dim, batch_size, encoder, decoder,
                 observation_dim=784,
                 learning_rate=1e-4,
                 optimizer=tf.compat.v1.train.RMSPropOptimizer,
                 observation_distribution="Gaussian", # or Gaussian
                 observation_std=0.01):

        self._latent_dim = latent_dim
        self._batch_size = batch_size
        self._encode = encoder
        self._decode = decoder
        self._observation_dim = observation_dim#输入，输出维度
        self._learning_rate = learning_rate
        self._optimizer = optimizer(learning_rate=self._learning_rate)
        self._observation_distribution = observation_distribution#隐变量Z分布
        self._observation_std = observation_std
        self._build()
    def _build(self):
        self.x=tf.placeholder(tf.float32,(None,self._observation_dim))
        with tf.variable_scope('encoder'):
            encoded=coders.vae_coding.fc_mnist_encoder(self.x,self._latent_dim)
            logvar=encoded[:,self._latent_dim:]
            self.mean=encoded[:,:self._latent_dim]
            epsilon=tf.random.normal((self._batch_size,self._latent_dim))
            self.z=epsilon*tf.sqrt(tf.exp(logvar))+self.mean
        with tf.variable_scope('decoder'):
            self.obs_mean=coders.vae_coding.fc_mnist_decoder(self.z,self._observation_dim)
        with tf.variable_scope('loss'):
            objective=self._gaussian_log_likelihood(self.x,self.obs_mean,self._observation_std)
            kl=self._kl_diagnormal_stdnormal(self.mean,logvar)
            self._loss=(objective+kl)/self._batch_size
            self._train=self._optimizer.minimize(self._loss)
        self._sess=tf.Session()
        self._sess.run(tf.global_variables_initializer())
    @staticmethod
    def _kl_diagnormal_stdnormal(mu, log_var):
        '''
        :param mu: 均值
        :param log_var:方差的对数
        :return: N(mu,log(var))和N(0,1)的K-L散度
        '''
        var = tf.exp(log_var)
        kl = 0.5 * tf.reduce_sum(tf.square(mu) + var - 1. - log_var)
        return kl

    @staticmethod
    def _gaussian_log_likelihood(targets, mean, std):
        se = 0.5 * tf.reduce_sum(tf.square(targets - mean)) / (2 * tf.square(std)) + tf.log(std)
        return se

    @staticmethod
    def _bernoulli_log_likelihood(targets, outputs, eps=1e-8):
        log_like = -tf.reduce_sum(targets * tf.log(outputs + eps)
                                  + (1. - targets) * tf.log((1. - outputs) + eps))
        return log_like
    def update(self,x):
        loss,_=self._sess.run([self._loss,self._train],feed_dict={self.x:x})
        return loss
    def z2x(self,z):
        x = self._sess.run([self.obs_mean], feed_dict={self.z: z})
        return x
    def x2z(self,x):
        mean = self._sess.run([self.mean], feed_dict={self.x: x})

        return np.asarray(mean).reshape(-1, self._latent_dim)
