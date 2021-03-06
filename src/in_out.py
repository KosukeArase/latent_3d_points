import six
import warnings
import numpy as np
import os
import os.path as osp
import re
import glob
import pickle
import functools
from six.moves import cPickle
from multiprocessing import Pool

from external.python_plyfile.plyfile import PlyElement, PlyData


def create_dir(dir_path):
    ''' Creates a directory (or nested directories) if they don't exist.
    '''
    if not osp.exists(dir_path):
        os.makedirs(dir_path)

    return dir_path


def pickle_data(file_name, *args):
    '''Using (c)Pickle to save multiple python objects in a single file.
    '''
    myFile = open(file_name, 'wb')
    cPickle.dump(len(args), myFile, protocol=2)
    for item in args:
        cPickle.dump(item, myFile, protocol=2)
    myFile.close()


def unpickle_data(file_name):
    '''Restore data previously saved with pickle_data().
    '''
    inFile = open(file_name, 'rb')
    size = cPickle.load(inFile)
    for _ in xrange(size):
        yield cPickle.load(inFile)
    inFile.close()


def files_in_subdirs(top_dir, search_pattern):
    regex = re.compile(search_pattern)
    for path, _, files in os.walk(top_dir):
        for name in files:
            full_name = osp.join(path, name)
            if regex.search(full_name):
                yield full_name


def load_txt(file_name, n_points=2048, with_faces=False, with_color=False):
    n_features = 6 if with_color else 3

    if with_faces:
        raise NotImplementedError
    else:
        with open(file_name, 'r') as f:
            lines = f.readlines()

    if len(lines) < n_points:
        print(len(lines))
        return None
    else:
        points = np.array([list(map(float, line.split()[:n_features])) for line in lines])
        points = points[np.random.choice(points.shape[0], n_points, replace = False)]

        mean = np.mean(points, axis=0)
        std = np.std(points, axis=0)
        points = (points - mean) / std

        return points


def pc_loader(f_name, n_points=2048, with_color=False):
    ''' loads a point-cloud saved under ShapeNet's "standar" folder scheme: 
    i.e. /syn_id/model_name.ply
    '''

    tokens = f_name.split('.')[1].split('/')
    model_name = '_'.join([tokens[2], tokens[3], tokens[5]])
    class_name = tokens[5].split('_')[0]

    try:
        points = load_txt(f_name, n_points, with_color=with_color)
    except Exception as e:
        print('error in pc_loader while proccesing', f_name, e)
        points = None

    return points, model_name, class_name


def load_all_point_clouds_under_folder(top_dir, class_name, n_threads=20, n_points=2048, with_color=False, verbose=False):
    if '6' in top_dir:
        data_type = 'test'
    elif '4' in top_dir:
        data_type = 'train_small'
    elif '1-5' in top_dir:
        data_type = 'train'
    else:
        print(top_dir, 'is an invalid dir')
        raise NotImplementedError

    glob_file = './s3dis/filenames/{}_{}.pkl'.format(data_type, class_name)

    if os.path.exists(glob_file):
        with open(glob_file, 'r') as f:
            file_names = pickle.load(f)
            print('Data paths are loaded.')
    else:
        if 'all' in class_name:
            class_names = ['table', 'chair', 'sofa', 'bookcase', 'board']
            file_names = []
            for name in class_names:
                file_names += glob.glob(top_dir.format(name))

            if class_name == 'all_w_clutter':
                file_names += glob.glob(top_dir.format('clutter'))
        else:
            file_names = glob.glob(top_dir.format(class_name))

        with open(glob_file, 'w') as f:
            pickle.dump(file_names, f)
            print('Data paths are created.')

    print('Loading {} data'.format(len(file_names)))

    pclouds, model_ids, syn_ids = load_point_clouds_from_filenames(file_names, n_threads, loader=pc_loader, n_points=n_points, with_color=with_color, verbose=verbose)
    return PointCloudDataSet(pclouds, labels=model_ids, init_shuffle=False)


def load_point_clouds_from_filenames(file_names, n_threads, loader, n_points=2048, with_color=False, verbose=False):
    n_features = 6 if with_color else 3 # XYZ or XYZRGB

    pclouds = np.empty([len(file_names), n_points, n_features], dtype=np.float32)

    model_names = np.empty([len(file_names)], dtype=object)
    class_ids = np.empty([len(file_names)], dtype=object)
    pool = Pool(n_threads)
    skipped = 0

    for i, data in enumerate(pool.imap(functools.partial(loader, n_points=n_points, with_color=with_color), file_names)):
        if data[0] is None:
            print(data[1:], "was skipped since it doesn't have enough points")
            skipped += 1
            continue
        else:
            pclouds[i-skipped, :, :], model_names[i-skipped], class_ids[i-skipped] = data

    pool.close()
    pool.join()

    pclouds = pclouds[:len(file_names)-skipped]
    model_names = model_names[:len(file_names)-skipped]
    class_ids = class_ids[:len(file_names)-skipped]

    if len(np.unique(model_names)) != len(pclouds):
        warnings.warn('Point clouds with the same model name were loaded.')

    if verbose:
        print('{0} pclouds were loaded. They belong in {1} shape-classes.'.format(len(pclouds), len(np.unique(class_ids))))

    return pclouds, model_names, class_ids


class PointCloudDataSet(object):
    '''
    See https://github.com/tensorflow/tensorflow/blob/a5d8217c4ed90041bea2616c14a8ddcf11ec8c03/tensorflow/examples/tutorials/mnist/input_data.py
    '''

    def __init__(self, point_clouds, noise=None, labels=None, copy=True, init_shuffle=True):
        '''Construct a DataSet.
        Args:
            init_shuffle, shuffle data before first epoch has been reached.
        Output:
            original_pclouds, labels, (None or Feed) # TODO Rename
        '''

        self.num_examples = point_clouds.shape[0]
        self.n_points = point_clouds.shape[1]

        if labels is not None:
            assert point_clouds.shape[0] == labels.shape[0], ('points.shape: %s labels.shape: %s' % (point_clouds.shape, labels.shape))
            if copy:
                self.labels = labels.copy()
            else:
                self.labels = labels

        else:
            self.labels = np.ones(self.num_examples, dtype=np.int8)

        if noise is not None:
            assert (type(noise) is np.ndarray)
            if copy:
                self.noisy_point_clouds = noise.copy()
            else:
                self.noisy_point_clouds = noise
        else:
            self.noisy_point_clouds = None

        if copy:
            self.point_clouds = point_clouds.copy()
        else:
            self.point_clouds = point_clouds

        self.epochs_completed = 0
        self._index_in_epoch = 0
        if init_shuffle:
            self.shuffle_data()

    def shuffle_data(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        perm = np.arange(self.num_examples)
        np.random.shuffle(perm)
        self.point_clouds = self.point_clouds[perm]
        self.labels = self.labels[perm]
        if self.noisy_point_clouds is not None:
            self.noisy_point_clouds = self.noisy_point_clouds[perm]
        return self

    def next_batch(self, batch_size, seed=None):
        '''Return the next batch_size examples from this data set.
        '''
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self.num_examples:
            self.epochs_completed += 1  # Finished epoch.
            self.shuffle_data(seed)
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch

        if self.noisy_point_clouds is None:
            return self.point_clouds[start:end], self.labels[start:end], None
        else:
            return self.point_clouds[start:end], self.labels[start:end], self.noisy_point_clouds[start:end]

    def full_epoch_data(self, shuffle=True, seed=None):
        '''Returns a copy of the examples of the entire data set (i.e. an epoch's data), shuffled.
        '''
        if shuffle and seed is not None:
            np.random.seed(seed)
        perm = np.arange(self.num_examples)  # Shuffle the data.
        if shuffle:
            np.random.shuffle(perm)
        pc = self.point_clouds[perm]
        lb = self.labels[perm]
        ns = None
        if self.noisy_point_clouds is not None:
            ns = self.noisy_point_clouds[perm]
        return pc, lb, ns
