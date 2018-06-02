'''
Created on November 26, 2017

@author: optas
'''

import numpy as np
import os.path as osp
from numpy.linalg import norm

from ae_templates import mlp_architecture_tl_net
from in_out import create_dir, pickle_data, unpickle_data


class Configuration():
    def __init__(self, n_input, encoder, decoder, embedder, encoder_args={}, decoder_args={}, embedder_args={},
                 training_epochs=200, batch_size=10, learning_rate=0.001, denoising=False,
                 saver_step=None, train_dir=None, z_rotate=False, input_color=False, output_color=False, loss='chamfer', gauss_augment=None,
                 saver_max_to_keep=None, loss_display_step=1, debug=False,
                 n_z=None, n_output=None, latent_vs_recon=1.0, consistent_io=None):

        # Parameters for any AE
        self.n_input = n_input
        self.is_denoising = denoising
        self.loss = loss.lower()
        self.decoder = decoder
        self.encoder = encoder
        self.embedder = embedder
        self.encoder_args = encoder_args
        self.decoder_args = decoder_args
        self.embedder_args = embedder_args

        # Training related parameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss_display_step = loss_display_step
        self.saver_step = saver_step
        self.train_dir = train_dir
        self.gauss_augment = gauss_augment
        self.z_rotate = z_rotate
        self.input_color = input_color
        self.output_color = output_color
        self.saver_max_to_keep = saver_max_to_keep
        self.training_epochs = training_epochs
        self.debug = debug

        # Used in VAE
        self.latent_vs_recon = np.array([latent_vs_recon], dtype=np.float32)[0]
        self.n_z = n_z

        # Used in AP
        if n_output is None:
            self.n_output = n_input
        else:
            self.n_output = n_output

        self.consistent_io = consistent_io

    def exists_and_is_not_none(self, attribute):
        return hasattr(self, attribute) and getattr(self, attribute) is not None

    def __str__(self):
        keys = self.__dict__.keys()
        vals = self.__dict__.values()
        index = np.argsort(keys)
        res = ''
        for i in index:
            if callable(vals[i]):
                v = vals[i].__name__
            else:
                v = str(vals[i])
            res += '%30s: %s\n' % (str(keys[i]), v)
        return res

    def save(self, file_name):
        pickle_data(file_name + '.pickle', self)
        with open(file_name + '.txt', 'w') as fout:
            fout.write(self.__str__())

    @staticmethod
    def load(file_name):
        return unpickle_data(file_name + '.pickle').next()


def get_conf(train_params):
    class_name = train_params['class_name']
    top_out_dir = './data/'          # Use to save Neural-Net check-points etc.
    experiment_name = '_'.join([class_name, train_params['experiment_name']])
    ckpt_path = '/'.join([train_params['experiment_name'], class_name])
    n_pc_points = train_params['n_pc_points']
    bneck_size = 128                  # Bottleneck-AE size
    ae_loss = 'emd'                   # Loss to optimize: 'emd' or 'chamfer'

    n_input_feat = 6 if train_params['input_color'] else 3
    n_output_feat = 6 if train_params['output_color'] else 3

    encoder, decoder, embedder, enc_args, dec_args, emb_args = mlp_architecture_tl_net(n_pc_points, bneck_size, n_output_feat=n_output_feat)
    train_dir = create_dir(osp.join(top_out_dir, ckpt_path))

    conf = Configuration(n_input = [n_pc_points, n_input_feat],
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


def get_visible_points(points, org=False):
    n_points = int(len(points)/2)
    ymean = np.mean(points[:, 1])
    points = points[points[:, 1] > ymean]

    points = points[np.random.choice(points.shape[0], n_points)] # allow duplicate

    mean = np.mean(points, axis=0)
    std = np.std(points, axis=0)
    norm_points = (points - mean) / std

    if org:
        return norm_points, points
    else:
        return norm_points


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
