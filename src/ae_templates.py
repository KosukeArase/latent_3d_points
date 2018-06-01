'''
Created on September 2, 2017

@author: optas
'''
import numpy as np

from encoders_decoders import encoder_with_convs_and_symmetry, decoder_with_fc_only, embedder_with_convs_and_symmetry


def mlp_architecture_tl_net(n_pc_points, bneck_size, bneck_post_mlp=False):
    ''' Single class experiments.
    '''
    if n_pc_points != 2048:
        raise ValueError()

    encoder = encoder_with_convs_and_symmetry
    decoder = decoder_with_fc_only
    embedder = embedder_with_convs_and_symmetry

    n_input = [n_pc_points, 3]

    encoder_args = {'n_filters': [64, 128, 128, 256, bneck_size],
                    'filter_sizes': [1],
                    'strides': [1],
                    'b_norm': True,
                    'scope': 'encoder',
                    'verbose': True
                    }

    decoder_args = {'layer_sizes': [256, 256, np.prod(n_input)],
                    'b_norm': False,
                    'b_norm_finish': False,
                    'scope': 'decoder',
                    'verbose': True
                    }

    embedder_args = {'n_filters': [64, 128, 128, 256, bneck_size],
                    'filter_sizes': [1],
                    'strides': [1],
                    'b_norm': True,
                    'scope': 'embedder',
                    'verbose': True
                    }

    if bneck_post_mlp:
        encoder_args['n_filters'].pop()
        decoder_args['layer_sizes'][0] = bneck_size

    return encoder, decoder, embedder, encoder_args, decoder_args, embedder_args


def mlp_architecture_ala_iclr_18(n_pc_points, bneck_size, bneck_post_mlp=False):
    ''' Single class experiments.
    '''
    if n_pc_points != 2048:
        raise ValueError()

    encoder = encoder_with_convs_and_symmetry
    decoder = decoder_with_fc_only

    n_input = [n_pc_points, 3]

    encoder_args = {'n_filters': [64, 128, 128, 256, bneck_size],
                    'filter_sizes': [1],
                    'strides': [1],
                    'b_norm': True,
                    'verbose': True
                    }

    decoder_args = {'layer_sizes': [256, 256, np.prod(n_input)],
                    'b_norm': False,
                    'b_norm_finish': False,
                    'verbose': True
                    }

    if bneck_post_mlp:
        encoder_args['n_filters'].pop()
        decoder_args['layer_sizes'][0] = bneck_size

    return encoder, decoder, encoder_args, decoder_args


def default_train_params(single_class=True):
    params = {'batch_size': 50,
              'training_epochs': 2000,
              'denoising': False,
              'learning_rate': 0.0005,
              'z_rotate': False,
              'saver_step': 10,
              'loss_display_step': 1
              }

    if not single_class:
        params['z_rotate'] = True
        params['training_epochs'] = 1000

    return params
