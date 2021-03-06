'''
Created on September 2, 2017

@author: optas
'''
import numpy as np

from encoders_decoders import encoder_with_convs_and_symmetry, decoder_with_fc_only, embedder_with_convs_and_symmetry


def mlp_architecture_tl_net(n_pc_points, bneck_size, n_output_feat=3, bneck_post_mlp=False):
    ''' Single class experiments.
    '''

    encoder = encoder_with_convs_and_symmetry
    decoder = decoder_with_fc_only
    embedder = embedder_with_convs_and_symmetry

    n_output = [n_pc_points, n_output_feat]

    encoder_args = {'n_filters': [64, 128, 128, 256, bneck_size],
                    'filter_sizes': [1],
                    'strides': [1],
                    'b_norm': True,
                    'scope': 'encoder',
                    'verbose': True
                    }

    decoder_args = {'layer_sizes': [256, 256, np.prod(n_output)],
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
