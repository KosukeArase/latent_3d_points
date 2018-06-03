import os
import pickle
import argparse
import matplotlib
import numpy as np

from src.point_net_ae import PointNetAutoEncoder
from src.in_out import load_all_point_clouds_under_folder
from src.tf_utils import reset_tf_graph
from src.evaluation_metrics import minimum_mathing_distance, jsd_between_point_cloud_sets, coverage
from src.general_utils import get_conf

def parse_args():
    params = {
          'n_pc_points': 2048,
          'batch_size': 50,
          'training_epochs': 1000,
          'denoising': False,
          'gauss_augment': False,
          'learning_rate': 0.002,
          'z_rotate': False,
          'saver_step': 100,
          'output_color': False,
          'loss_display_step': 1,
          }

    parser = argparse.ArgumentParser(
            prog="Visualizing PC2PC TL network",
            usage="python visualize.py [class_name] [options]", #Usage
            add_help = True
            )

    parser.add_argument("class_name", help="Name of class (e.g. 'chair' or 'all' or 'all_w_clutter')")
    parser.add_argument("experiment_name", help="Name of experiment")
    parser.add_argument("-n", "--n_points", help="Number of input points", default=2048, type=int)
    parser.add_argument("-l", "--loss", help="Loss type", default='emd', type=str)
    parser.add_argument("-c", "--input_color", action='store_true', help="Add color to input feature")

    args = parser.parse_args()

    params['class_name'] = args.class_name
    params['experiment_name'] = args.experiment_name
    params['n_pc_points'] = args.n_points
    params['input_color'] = args.input_color
    params['ae_loss'] = args.loss

    return params


def main():
    params = parse_args()
    conf = get_conf(params)

    conf.gauss_augment = False
    conf.z_rotate = False

    ckpt_path = os.path.join('./data/', params['experiment_name'], params['class_name'])
    epoch = conf.training_epochs

    reset_tf_graph()
    ae = PointNetAutoEncoder(conf.experiment_name, conf)
    ae.restore_model(ckpt_path, epoch, verbose=True)

    test_dir = './s3dis/Area_6/*/Annotations/{}_*.txt'
    all_pc_data = load_all_point_clouds_under_folder(test_dir, params['class_name'], n_points=conf.n_input[0], with_color=conf.input_color, n_threads=20, verbose=True)

    ae_recon, tl_recon, data_loss, labels, gt, gt_tl = ae.evaluate(all_pc_data, conf) # gt_tl is not normalized to compare with gt

    if conf.input_color:
        gt = gt[:, :, :3]

    ae_loss = conf.loss  # Which distance to use for the matchings.
    assert ae_loss in ['emd', 'chamfer']

    if ae_loss == 'emd':
        use_EMD = True
    else:
        use_EMD = False  # Will use Chamfer instead.

    batch_size = 100     # Find appropriate number that fits in GPU.
    normalize = True     # Matched distances are divided by the number of points of thepoint-clouds.

    ae_mmd, ae_matched_dists = minimum_mathing_distance(ae_recon, gt, batch_size, normalize=normalize, use_EMD=use_EMD)
    tl_mmd, tl_matched_dists = minimum_mathing_distance(tl_recon, gt, batch_size, normalize=normalize, use_EMD=use_EMD)

    ae_cov, ae_matched_ids = coverage(ae_recon, gt, batch_size, normalize=normalize, use_EMD=use_EMD)
    tl_cov, tl_matched_ids = coverage(tl_recon, gt, batch_size, normalize=normalize, use_EMD=use_EMD)
    
    ae_jsd = jsd_between_point_cloud_sets(ae_recon, gt, resolution=28)
    tl_jsd = jsd_between_point_cloud_sets(tl_recon, gt, resolution=28)

    result_file = os.path.join('result', params['experiment_name'] + '_' + params['class_name'] + '.txt')

    with open(result_file, 'w') as f:
        f.write('ae_mmd' + str(ae_mmd) + '\n')
        f.write('tl_mmd' + str(tl_mmd) + '\n')
        # f.write('ae_matched_dists' + str(ae_matched_dists) + '\n')
        f.write('ae_cov' + str(ae_cov) + '\n')
        f.write('tl_cov' + str(tl_cov) + '\n')
        # f.write('ae_matched_ids' + str(ae_matched_ids) + '\n')
        f.write('ae_jsd' + str(ae_jsd) + '\n')
        f.write('tl_jsd' + str(tl_jsd) + '\n')


if __name__ == '__main__':
    main()
