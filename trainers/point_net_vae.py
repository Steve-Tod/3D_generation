'''
Created on January 26, 2017
Modified on June 14, 2019

@author: optas
@modify: Tod
'''

import time
import tensorflow as tf
import os.path as osp

from tflearn.layers.conv import conv_1d
from tflearn.layers.core import fully_connected

from utils.dirs import create_dir
from .autoencoder import AutoEncoder
from utils.utils import apply_augmentations

try:
    from models.structural_losses.tf_nndistance import nn_distance
    from models.structural_losses.tf_approxmatch import approx_match, match_cost
except:
    print(
        'External Losses (Chamfer-EMD) cannot be loaded. Please install them first.'
    )


class PointNetVariationalAutoEncoder(AutoEncoder):
    '''
    An Auto-Encoder for point-clouds.
    '''

    def __init__(self, name, configuration, graph=None):
        c = configuration
        self.configuration = c

        AutoEncoder.__init__(self, name, graph, configuration)
        self.global_step = 0

        with tf.variable_scope(name):
            self.z_mean, self.z_std = c.encoder(self.x, **c.encoder_args)
            eps = tf.random_normal(tf.shape(self.z_std),
                                   dtype=tf.float32,
                                   mean=0.,
                                   stddev=1.0,
                                   name='epsilon')
            self.z = self.z_mean + tf.exp(self.z_std / 2) * eps
            self.bottleneck_size = int(self.z.get_shape()[1])
            layer = c.decoder(self.z, **c.decoder_args)

            if c.exists_and_is_not_none('close_with_tanh'):
                layer = tf.nn.tanh(layer)

            self.x_reconstr = tf.reshape(
                layer, [-1, self.n_output[0], self.n_output[1]])

            self.saver = tf.train.Saver(tf.global_variables(),
                                        max_to_keep=c.saver_max_to_keep)

            self._create_loss()
            self._setup_optimizer()

            # GPU configuration
            if hasattr(c, 'allow_gpu_growth'):
                growth = c.allow_gpu_growth
            else:
                growth = True

            if growth:
                print('allow growth')
            else:
                print('not allow growth')
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = growth

            # Summaries
            # tf.summary.scalar('loss', self.loss)
            self.merged_summaries = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(
                osp.join(configuration.train_dir, 'summaries'), self.graph)

            # Initializing the tensor flow variables
            self.init = tf.global_variables_initializer()

            # Launch the session
            self.sess = tf.Session(config=config)
            self.sess.run(self.init)

    def _create_loss(self):
        c = self.configuration

        if c.loss == 'chamfer':
            cost_p1_p2, _, cost_p2_p1, _ = nn_distance(self.x_reconstr,
                                                       self.gt)
            self.recon_loss = tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(
                cost_p2_p1)
        elif c.loss == 'emd':
            match = approx_match(self.x_reconstr, self.gt)
            self.recon_loss = tf.reduce_mean(
                match_cost(self.x_reconstr, self.gt, match))

        reg_losses = self.graph.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES)
        if c.exists_and_is_not_none('w_reg_alpha'):
            w_reg_alpha = c.w_reg_alpha
        else:
            w_reg_alpha = 1.0

        if c.exists_and_is_not_none('latent_vs_recon'):
            w_kld = c.latent_vs_recon
        else:
            w_kld = 1.0

        self.reg_loss = 0.0
        for rl in reg_losses:
            self.reg_loss += w_reg_alpha * rl

        # KL Divergence
        self.kl_div_loss = 1 + self.z_std - tf.square(self.z_mean) - tf.exp(
            self.z_std)
        self.kl_div_loss = -0.5 * tf.reduce_sum(self.kl_div_loss, 1)
        self.kl_div_loss = tf.reduce_mean(self.kl_div_loss) * w_kld

        self.loss = self.recon_loss + self.reg_loss + self.kl_div_loss
        with tf.name_scope('performance'):
            tf.summary.scalar('recon_loss', self.recon_loss)
            tf.summary.scalar('reg_loss', self.reg_loss)
            tf.summary.scalar('kl_div_loss', self.kl_div_loss)

    def _setup_optimizer(self):
        c = self.configuration
        self.lr = c.learning_rate
        if hasattr(c, 'exponential_decay'):
            self.lr = tf.train.exponential_decay(c.learning_rate,
                                                 self.epoch,
                                                 c.decay_steps,
                                                 decay_rate=0.5,
                                                 staircase=True,
                                                 name="learning_rate_decay")
            self.lr = tf.maximum(self.lr, 1e-5)
            tf.summary.scalar('learning_rate', self.lr)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_step = self.optimizer.minimize(self.loss)

    def _single_epoch_train(self, train_data, configuration, only_fw=False):
        n_examples = train_data.num_examples
        epoch_loss = 0.
        batch_size = configuration.batch_size
        n_batches = int(n_examples / batch_size)
        start_time = time.time()

        if only_fw:
            fit = self.reconstruct
        else:
            fit = self.partial_fit

        # Loop over all batches
        for iteration in range(n_batches):

            if self.is_denoising:
                original_data, _, batch_i = train_data.next_batch(batch_size)
                if batch_i is None:  # In this case the denoising concern only the augmentation.
                    batch_i = original_data
            else:
                batch_i, _, _ = train_data.next_batch(batch_size)

            batch_i = apply_augmentations(
                batch_i, configuration)  # This is a new copy of the batch.

            if self.is_denoising:
                _, loss, summary = fit(batch_i,
                                       original_data,
                                       need_summary=True)
            else:
                _, loss, summary = fit(batch_i, need_summary=True)

            # Compute average loss
            epoch_loss += loss
            if iteration % configuration.loss_summary_step == 0 and iteration != 0:
                self.train_writer.add_summary(summary, self.global_step)
            self.global_step += 1
        epoch_loss /= n_batches
        duration = time.time() - start_time

        if configuration.loss == 'emd':
            epoch_loss /= len(train_data.point_clouds[0])

        return epoch_loss, duration

    def gradient_of_input_wrt_loss(self, in_points, gt_points=None):
        if gt_points is None:
            gt_points = in_points
        return self.sess.run(tf.gradients(self.loss, self.x),
                             feed_dict={
                                 self.x: in_points,
                                 self.gt: gt_points
                             })
