import tensorflow as tf
from coders.vae_coding import fc_mnist_decoder,fc_mnist_encoder
from models.myvae import myvae
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm
import numpy as np
from plots.grid_plots import show_latent_scatter,show_samples
def main():
    flags = tf.flags
    flags.DEFINE_integer("latent_dim", 2, "Dimension of latent space.")
    flags.DEFINE_integer("batch_size", 128, "Batch size.")
    flags.DEFINE_integer("epochs", 100, "As it said")
    flags.DEFINE_integer("updates_per_epoch", 100, "Really just can set to 1 if you don't like mini-batch.")
    flags.DEFINE_string("data_dir", 'mnist', "Tensorflow demo data download position.")
    FLAGS = flags.FLAGS
    kwargs = {
        'latent_dim': FLAGS.latent_dim,
        'batch_size': FLAGS.batch_size,
        'encoder': fc_mnist_encoder,
        'decoder': fc_mnist_decoder
    }
    vae = myvae(**kwargs)
    mnist = input_data.read_data_sets(train_dir=FLAGS.data_dir)
    tbar = tqdm(range(FLAGS.epochs))
    for epoch in tbar:
        training_loss = 0.

        for _ in range(FLAGS.updates_per_epoch):
            x, _ = mnist.train.next_batch(FLAGS.batch_size)#x为sample,_为label
            loss = vae.update(x)
            training_loss += loss

        training_loss /= FLAGS.updates_per_epoch
        s = "Loss: {:.4f}".format(training_loss)
        tbar.set_description(s)

    '用训练好的模型产生新的样本'
    z = np.random.normal(size=[FLAGS.batch_size, FLAGS.latent_dim])
    samples = vae.z2x(z)[0]
    show_samples(samples, 10, 10, [28, 28], name='samples')
    show_latent_scatter(vae, mnist, name='latent')

    vae.save_generator('weights/vae_mnist/generator')
if __name__ == '__main__':
    main()
