'''
Created on January 26, 2017

@author: optas
'''

import time
import tensorflow as tf
import os.path as osp

from tflearn.layers.conv import conv_1d
from tflearn.layers.core import fully_connected

from autoencoder import AutoEncoder
from general_utils import apply_augmentations

try:    
    from external.structural_losses.tf_nndistance import nn_distance
    from external.structural_losses.tf_approxmatch import approx_match, match_cost
except:
    print('External Losses (Chamfer-EMD) cannot be loaded. Please install them first.')
    

class PointNetAutoEncoder(AutoEncoder):
    '''
    An Auto-Encoder for point-clouds.
    '''

    def __init__(self, name, configuration, graph=None):
        c = configuration
        self.configuration = c

        AutoEncoder.__init__(self, name, graph, configuration)

        with tf.variable_scope(name):
            self.z = c.encoder(self.x, **c.encoder_args)
            self.vz = c.embedder(self.vx, **c.embedder_args)

            self.bottleneck_size = int(self.z.get_shape()[1])
            layer = c.decoder(self.z, **c.decoder_args)
            c.decoder_args['reuse'] = True
            vlayer = c.decoder(self.vz, **c.decoder_args)
            c.decoder_args['reuse'] = False
            
            if c.exists_and_is_not_none('close_with_tanh'):
                layer = tf.nn.tanh(layer)
                vlayer = tf.nn.tanh(vlayer)


            self.x_reconstr = tf.reshape(layer, [-1, self.n_output[0], self.n_output[1]])
            self.vx_reconstr = tf.reshape(vlayer, [-1, self.n_output[0], self.n_output[1]])
            
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=c.saver_max_to_keep)

            self._create_loss()
            self._setup_optimizer()

            # GPU configuration
            if hasattr(c, 'allow_gpu_growth'):
                growth = c.allow_gpu_growth
            else:
                growth = True

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = growth

            # Summaries
            self.merged_summaries = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(osp.join(configuration.train_dir, 'summaries'), self.graph)

            # Initializing the tensor flow variables
            self.init = tf.global_variables_initializer()

            # Launch the session
            self.sess = tf.Session(config=config)
            self.sess.run(self.init)

    def _create_loss(self):
        lambda_x = 1.0
        lambda_z = 1.0
        c = self.configuration

        n_output_feat = c.n_output[1]
        assert n_output_feat in [3, 6]

        if c.loss == 'chamfer':
            cost_p1_p2, _, cost_p2_p1, _ = nn_distance(self.x_reconstr[:, :, :n_output_feat], self.gt[:, :, :n_output_feat])
            self.x_loss = tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1)
        elif c.loss == 'emd':
            match = approx_match(self.x_reconstr[:, :, :n_output_feat], self.gt[:, :, :n_output_feat])
            self.x_loss = tf.reduce_mean(match_cost(self.x_reconstr[:, :, :n_output_feat], self.gt[:, :, :n_output_feat], match))

        z_stopped = tf.stop_gradient(self.z)
        self.vz_loss = tf.nn.l2_loss(self.vz - z_stopped)
        self.z_total_loss = tf.nn.l2_loss(self.vz - self.z)

        self.x_loss *= lambda_x
        self.vz_loss *= lambda_z
        self.z_total_loss *= lambda_z
        self.total_loss = self.x_loss + self.z_total_loss


        reg_losses = self.graph.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if c.exists_and_is_not_none('w_reg_alpha'):
            w_reg_alpha = c.w_reg_alpha
        else:
            w_reg_alpha = 1.0

        for rl in reg_losses:
            self.x_loss += (w_reg_alpha * rl)
            self.vz_loss += (w_reg_alpha * rl)
            self.total_loss += (w_reg_alpha * rl)

    def _setup_optimizer(self):
        c = self.configuration
        self.lr = c.learning_rate
        if hasattr(c, 'exponential_decay'):
            self.lr = tf.train.exponential_decay(c.learning_rate, self.epoch, c.decay_steps, decay_rate=0.5, staircase=True, name="learning_rate_decay")
            self.lr = tf.maximum(self.lr, 1e-5)
            tf.summary.scalar('learning_rate', self.lr)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_step1 = self.optimizer.minimize(self.x_loss)
        self.train_step2 = self.optimizer.minimize(self.vz_loss)
        self.train_step3 = self.optimizer.minimize(self.total_loss)

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
        for _ in xrange(n_batches):

            if self.is_denoising:
                original_data, _, batch_i = train_data.next_batch(batch_size)
                if batch_i is None:  # In this case the denoising concern only the augmentation.
                    batch_i = original_data
            else:
                original_data, _, _ = train_data.next_batch(batch_size)
                batch_i = original_data


            batch, batch_vis = apply_augmentations(batch_i, configuration) # This is a new copy of the batch.

            if self.is_denoising:
                _, loss = fit([batch, batch_vis], original_data)
            else:
                _, loss = fit([batch, batch_vis])

            # Compute average loss
            epoch_loss += loss
        epoch_loss /= n_batches
        duration = time.time() - start_time
        
        if configuration.loss == 'emd':
            epoch_loss /= len(train_data.point_clouds[0])
        
        return epoch_loss, duration

    # def gradient_of_input_wrt_loss(self, in_points, gt_points=None):
    #     if gt_points is None:
    #         gt_points = in_points
    #     return self.sess.run(tf.gradients(self.loss, self.x), feed_dict={self.x: in_points, self.gt: gt_points})
