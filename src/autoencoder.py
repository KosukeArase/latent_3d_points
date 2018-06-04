'''
Created on February 2, 2017

@author: optas
'''

import warnings
import os.path as osp
import tensorflow as tf
import numpy as np

from tflearn import is_training

from in_out import create_dir, pickle_data, unpickle_data
from general_utils import apply_augmentations, iterate_in_chunks, get_visible_points
from neural_net import Neural_Net

model_saver_id = 'models.ckpt'


class AutoEncoder(Neural_Net):
    '''Basis class for a Neural Network that implements an Auto-Encoder in TensorFlow.
    '''

    def __init__(self, name, graph, configuration):
        Neural_Net.__init__(self, name, graph)
        self.is_denoising = configuration.is_denoising
        self.n_input = configuration.n_input
        self.n_output = configuration.n_output

        in_shape = [None] + self.n_input
        v_in_shape = [None] + [int(self.n_input[0]/2), self.n_input[1]]
        out_shape = [None] + self.n_output

        with tf.variable_scope(name):
            self.x = tf.placeholder(tf.float32, in_shape)
            self.vx = tf.placeholder(tf.float32, v_in_shape)
            if self.is_denoising:
                self.gt = tf.placeholder(tf.float32, out_shape)
            else:
                self.gt = self.x

    def restore_model(self, model_path, epoch, verbose=False):
        '''Restore all the variables of a saved auto-encoder model.
        '''
        self.saver.restore(self.sess, osp.join(model_path, model_saver_id + '-' + str(int(epoch))))

        if self.epoch.eval(session=self.sess) != epoch:
            warnings.warn('Loaded model\'s epoch doesn\'t match the requested one.')
        else:
            if verbose:
                print('Model restored in epoch {0}.'.format(epoch))

    def partial_fit(self, X, GT=None):
        '''Trains the model with mini-batches of input data.
        If GT is not None, then the reconstruction loss compares the output of the net that is fed X, with the GT.
        This can be useful when training for instance a denoising auto-encoder.
        Returns:
            The loss of the mini-batch.
            The reconstructed (output) point-clouds.
        '''
        is_training(True, session=self.sess)
        try:
            if GT is not None:
                _, loss, recon = self.sess.run((self.opt, self.loss, self.x_reconstr), feed_dict={self.x: X[0], self.vx: X[1], self.gt: GT})
            else:
                _, loss, recon = self.sess.run((self.opt, self.loss, self.x_reconstr), feed_dict={self.x: X[0], self.vx: X[1]})

            is_training(False, session=self.sess)
        except Exception:
            raise
        finally:
            is_training(False, session=self.sess)
        return recon, loss

    def reconstruct(self, X, GT=None, compute_loss=True):
        '''Use AE to reconstruct given data.
        GT will be used to measure the loss (e.g., if X is a noisy version of the GT)'''
        if compute_loss:
            loss = self.total_loss
        else:
            loss = tf.no_op()

        if GT is None:
            return self.sess.run((self.x_reconstr, self.vx_reconstr, loss), feed_dict={self.x: X[0], self.vx: X[1]})
        else:
            return self.sess.run((self.x_reconstr, self.vx_reconstr, loss), feed_dict={self.x: X[0], self.vx: X[1], self.gt: GT})

    # def transform(self, X):
    #     '''Transform data by mapping it into the latent space.'''
    #     return self.sess.run(self.z, feed_dict={self.x: X})

    # def interpolate(self, x, y, steps):
    #     ''' Interpolate between and x and y input vectors in latent space.
    #     x, y np.arrays of size (n_points, dim_embedding).
    #     '''
    #     in_feed = np.vstack((x, y))
    #     z1, z2 = self.transform(in_feed.reshape([2] + self.n_input))
    #     all_z = np.zeros((steps + 2, len(z1)))

    #     for i, alpha in enumerate(np.linspace(0, 1, steps + 2)):
    #         all_z[i, :] = (alpha * z2) + ((1.0 - alpha) * z1)

    #     return self.sess.run((self.x_reconstr), {self.z: all_z})

    def decode(self, z):
        if np.ndim(z) == 1:  # single example
            z = np.expand_dims(z, 0)
        return self.sess.run((self.x_reconstr), {self.z: z})

    def train(self, train_data, configuration, log_file=None, held_out_data=None):
        c = configuration
        stats = []

        print('==============\nFirst Training Stage\n==============')

        if c.saver_step is not None:
            create_dir(c.train_dir)

        for _ in xrange(c.training_epochs):
            loss, duration = self._single_epoch_train(train_data, c)
            epoch = int(self.sess.run(self.epoch.assign_add(tf.constant(1.0))))
            stats.append((epoch, loss, duration))

            if epoch % c.loss_display_step == 0:
                print("Epoch:", '%04d' % (epoch), 'training time (minutes)=', "{:.4f}".format(duration / 60.0), "loss=", "{:.9f}".format(loss))
                if log_file is not None:
                    log_file.write('%04d\t%.9f\t%.4f\n' % (epoch, loss, duration / 60.0))

            # Save the models checkpoint periodically.
            if c.saver_step is not None and (epoch % c.saver_step == 0 or epoch - 1 == 0):
                checkpoint_path = osp.join(c.train_dir, model_saver_id)
                self.saver.save(self.sess, checkpoint_path, global_step=self.epoch)

            if c.exists_and_is_not_none('summary_step') and (epoch % c.summary_step == 0 or epoch - 1 == 0):
                summary = self.sess.run(self.merged_summaries)
                self.train_writer.add_summary(summary, epoch)

            if held_out_data is not None and c.exists_and_is_not_none('held_out_step') and (epoch % c.held_out_step == 0):
                loss, duration = self._single_epoch_train(held_out_data, c, only_fw=True)
                print("Held Out Data :", 'forward time (minutes)=', "{:.4f}".format(duration / 60.0), "loss=", "{:.9f}".format(loss))
                if log_file is not None:
                    log_file.write('On Held_Out: %04d\t%.9f\t%.4f\n' % (epoch, loss, duration / 60.0))

            if epoch == int(c.training_epochs/4):
                print('==============\nSecond Training Stage\n==============')
                self.opt = self.train_step2
                self.loss = self.vz_loss

            elif epoch == int(c.training_epochs/2):
                print('==============\nThird Training Stage\n==============')
                self.opt = self.train_step3
                self.loss = self.total_loss

            elif epoch == int(c.training_epochs/4*3):
                print('==============\nFourth Training Stage\n==============')
                self.lr *= 0.1

        return stats

    def evaluate(self, in_data, configuration):
        n_examples = in_data.num_examples
        data_loss = 0.

        feed_data, ids, _ = in_data.full_epoch_data(shuffle=False)

        b = configuration.batch_size
        ae_recon = np.zeros([n_examples] + self.n_output)
        v_recon = np.zeros([n_examples] + self.n_output)

        for i in xrange(0, n_examples, b):
            feed_data_v, feed_data_v_org = np.split(np.array([get_visible_points(x, org=True) for x in feed_data[i:i + b]]), 2, axis=1) # [normalized_pcs, original_pcs]
            feed_data_v = np.squeeze(feed_data_v)
            ae_recon[i:i + b], v_recon[i:i + b], loss = self.reconstruct([feed_data[i:i + b], feed_data_v])

            # Compute average loss
            data_loss += (loss * b)

        data_loss /= float(n_examples)
        return ae_recon, v_recon, data_loss, ids, np.squeeze(feed_data), np.squeeze(feed_data_v_org)

    def evaluate_one_by_one(self, in_data, configuration):
        '''Evaluates every data point separately to recover the loss on it. Thus, the batch_size = 1 making it
        a slower than the 'evaluate' method.
        '''

        if self.is_denoising:
            original_data, ids, feed_data = in_data.full_epoch_data(shuffle=False)
            if feed_data is None:
                feed_data = original_data
            feed_data = apply_augmentations(feed_data, configuration)  # This is a new copy of the batch.
        else:
            original_data, ids, _ = in_data.full_epoch_data(shuffle=False)
            feed_data = apply_augmentations(original_data, configuration)

        n_examples = in_data.num_examples
        assert(len(original_data) == n_examples)

        feed_data = np.expand_dims(feed_data, 1)
        original_data = np.expand_dims(original_data, 1)
        reconstructions = np.zeros([n_examples] + self.n_output)
        losses = np.zeros([n_examples])

        for i in xrange(n_examples):
            if self.is_denoising:
                reconstructions[i], losses[i] = self.reconstruct(feed_data[i], original_data[i])
            else:
                reconstructions[i], losses[i] = self.reconstruct(feed_data[i])

        return reconstructions, losses, np.squeeze(feed_data), ids, np.squeeze(original_data)

    # def embedding_at_tensor(self, dataset, conf, feed_original=True, apply_augmentation=False, tensor_name='bottleneck'):
    #     '''
    #     Observation: the NN-neighborhoods seem more reasonable when we do not apply the augmentation.
    #     Observation: the next layer after latent (z) might be something interesting.
    #     tensor_name: e.g. model.name + '_1/decoder_fc_0/BiasAdd:0'
    #     '''
    #     batch_size = conf.batch_size
    #     original, ids, noise = dataset.full_epoch_data(shuffle=False)

    #     if feed_original:
    #         feed = original
    #     else:
    #         feed = noise
    #         if feed is None:
    #             feed = original

    #     feed_data = feed
    #     if apply_augmentation:
    #         feed_data = apply_augmentations(feed, conf)

    #     embedding = []
    #     if tensor_name == 'bottleneck':
    #         for b in iterate_in_chunks(feed_data, batch_size):
    #             embedding.append(self.transform(b.reshape([len(b)] + conf.n_input)))
    #     else:
    #         embedding_tensor = self.graph.get_tensor_by_name(tensor_name)
    #         for b in iterate_in_chunks(feed_data, batch_size):
    #             codes = self.sess.run(embedding_tensor, feed_dict={self.x: b.reshape([len(b)] + conf.n_input)})
    #             embedding.append(codes)

    #     embedding = np.vstack(embedding)
    #     return feed, embedding, ids
