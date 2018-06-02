'''
Created on November 26, 2017

@author: optas
'''

import numpy as np
import os.path as osp
from numpy.linalg import norm

from autoencoder import Configuration as Conf
from ae_templates import mlp_architecture_tl_net, default_train_params
from in_out import create_dir


def get_conf(class_name):
    top_out_dir = './data/'          # Use to save Neural-Net check-points etc.
    experiment_name = '{}_tl'.format(class_name)
    n_pc_points = 2048                # Number of points per model.
    bneck_size = 128                  # Bottleneck-AE size
    ae_loss = 'emd'                   # Loss to optimize: 'emd' or 'chamfer'

    train_params = default_train_params()

    n_input_feat = 6 if train_params['input_color'] else 3
    n_output_feat = 6 if train_params['output_color'] else 3

    encoder, decoder, embedder, enc_args, dec_args, emb_args = mlp_architecture_tl_net(n_pc_points, bneck_size, n_output_feat=n_output_feat)
    train_dir = create_dir(osp.join(top_out_dir, experiment_name))

    conf = Conf(n_input = [n_pc_points, n_input_feat],
            n_output = [n_pc_points, n_output_feat],
            loss = ae_loss,
            training_epochs = train_params['training_epochs'],
            batch_size = train_params['batch_size'],
            denoising = train_params['denoising'],
            learning_rate = train_params['learning_rate'],
            train_dir = train_dir,
            loss_display_step = train_params['loss_display_step'],
            saver_step = train_params['saver_step'],
            z_rotate = train_params['z_rotate'],
            input_color = train_params['input_color'],
            output_color = train_params['output_color'],
            encoder = encoder,
            decoder = decoder,
            embedder = embedder,
            encoder_args = enc_args,
            decoder_args = dec_args,
            embedder_args = emb_args
           )
    conf.experiment_name = experiment_name
    conf.held_out_step = 5   # How often to evaluate/print out loss on 
                             # held_out data (if they are provided in ae.train() ).
    conf.save(osp.join(train_dir, 'configuration'))

    return conf


def rand_rotation_matrix(deflection=1.0, seed=None):
    '''Creates a random rotation matrix.

    deflection: the magnitude of the rotation. For 0, no rotation; for 1, completely random
    rotation. Small deflection => small perturbation.

    DOI: http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
         http://blog.lostinmyterminal.com/python/2015/05/12/random-rotation-matrix.html
    '''
    if seed is not None:
        np.random.seed(seed)

    randnums = np.random.uniform(size=(3,))

    theta, phi, z = randnums

    theta = theta * 2.0 * deflection * np.pi    # Rotation about the pole (Z).
    phi = 0 #phi * 2.0 * np.pi     # For direction of pole deflection.
    z = 0 # z * 2.0 * deflection    # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z))

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.
    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M


def iterate_in_chunks(l, n):
    '''Yield successive 'n'-sized chunks from iterable 'l'.
    Note: last chunk will be smaller than l if n doesn't divide l perfectly.
    '''
    for i in xrange(0, len(l), n):
        yield l[i:i + n]

        
def add_gaussian_noise_to_pcloud(pcloud, mu=0, sigma=1):
    gnoise = np.random.normal(mu, sigma, pcloud.shape[0])
    gnoise = np.tile(gnoise, (3, 1)).T
    pcloud += gnoise
    return pcloud


def get_visible_points(points):
    n_points = int(len(points)/2)
    ymean = np.mean(points[:, 1])
    points = points[points[:, 1] > ymean]

    points = points[np.random.choice(points.shape[0], n_points)] # allow duplicate

    mean = np.mean(points, axis=0)
    std = np.std(points, axis=0)
    points = (points - mean) / std

    return points


def apply_augmentations(batch, conf):
    if conf.gauss_augment is not None or conf.z_rotate:
        batch = batch.copy()

    if conf.gauss_augment is not None:
        mu = conf.gauss_augment['mu']
        sigma = conf.gauss_augment['sigma']
        batch += np.random.normal(mu, sigma, batch.shape)

    if conf.z_rotate:
        r_rotation = rand_rotation_matrix()
        r_rotation[0, 2] = 0
        r_rotation[2, 0] = 0
        r_rotation[1, 2] = 0
        r_rotation[2, 1] = 0
        r_rotation[2, 2] = 1
        batch[:, :, :3] = batch[:, :, :3].dot(r_rotation)

    batch_viz = np.array([get_visible_points(x) for x in batch])
    return batch, batch_viz


def unit_cube_grid_point_cloud(resolution, clip_sphere=False):
    '''Returns the center coordinates of each cell of a 3D grid with resolution^3 cells,
    that is placed in the unit-cube.
    If clip_sphere it True it drops the "corner" cells that lie outside the unit-sphere.
    '''
    grid = np.ndarray((resolution, resolution, resolution, 3), np.float32)
    spacing = 1.0 / float(resolution - 1)
    for i in xrange(resolution):
        for j in xrange(resolution):
            for k in xrange(resolution):
                grid[i, j, k, 0] = i * spacing - 0.5
                grid[i, j, k, 1] = j * spacing - 0.5
                grid[i, j, k, 2] = k * spacing - 0.5

    if clip_sphere:
        grid = grid.reshape(-1, 3)
        grid = grid[norm(grid, axis=1) <= 0.5]

    return grid, spacing
