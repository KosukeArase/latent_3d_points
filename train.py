import os.path as osp

from src.point_net_ae import PointNetAutoEncoder
from src.in_out import PointCloudDataSet, load_all_point_clouds_under_folder
from src.tf_utils import reset_tf_graph
from src.general_utils import get_conf
from src.visualize import visualize

import argparse


def parse_args():
    params = {'class_name': None,
          'n_pc_points': 2048,
          'batch_size': 50,
          'training_epochs': 2000,
          'denoising': False,
          'learning_rate': 0.0005,
          'z_rotate': True,
          'saver_step': 100,
          'input_color':True,
          'output_color': False,
          'loss_display_step': 1,
          'experiment_name': 'tmp'
          }

    parser = argparse.ArgumentParser(
            prog="Training PC2PC TL network",
            usage="python main.py [class_name] [options]", #Usage
            add_help = True
            )

    parser.add_argument("class_name", help="Name of class (e.g. 'chair' or 'all' or 'all_w_clutter')")
    parser.add_argument("experiment_name", help="Name of experiment")
    parser.add_argument("-n", "--n_points", help="Number of input points", default=2048, type=int)
    parser.add_argument("-c", "--input_color", action='store_true', help="Add color to input feature")
    parser.add_argument("-v", "--visualize", action='store_true', help="Visualize the result")

    args = parser.parse_args()

    params['class_name'] = args.class_name
    params['experiment_name'] = args.experiment_name
    params['n_pc_points'] = args.n_points
    params['input_color'] = args.input_color

    return params


def main():
    params = parse_args()
    conf = get_conf(params)

    train_dir = './s3dis/Area_[1-5]/*/Annotations/{}_*.txt'
    # train_dir = './s3dis/Area_4/*/Annotations/{}_*.txt'
    test_dir = './s3dis/Area_6/*/Annotations/{}_*.txt'

    reset_tf_graph()
    ae = PointNetAutoEncoder(conf.experiment_name, conf)

    buf_size = 1 # Make 'training_stats' file to flush each output line regarding training.
    fout = open(osp.join(conf.train_dir, 'train_stats.txt'), 'a', buf_size)

    all_pc_data = load_all_point_clouds_under_folder(train_dir, params['class_name'], n_points=conf.n_input[0], with_color=conf.input_color, n_threads=20, verbose=True)

    print('Start training')
    train_stats = ae.train(all_pc_data, conf, log_file=fout)
    fout.close()

    print('Finish training')

    if visualize:
        print('Start visualizing')
        reset_tf_graph()
        ae = PointNetAutoEncoder(conf.experiment_name, conf)
        ae.restore_model(ckpt_path, epoch, verbose=True)

        ckpt_path = os.path.join('./data/', params['experiment_name'], params['class_name'])
        epoch = conf.training_epochs
        all_pc_data = load_all_point_clouds_under_folder(test_dir, params['class_name'], n_points=conf.n_input[0], with_color=conf.input_color, n_threads=20, verbose=True)
        all_pc_data.shuffle_data()

        feed_pc, feed_model_names, _ = all_pc_data.next_batch(10)
        feed_pc_v, feed_pc_v_org = np.split(np.array([get_visible_points(x, org=True) for x in feed_pc]), 2, axis=1) # [normalized_pcs, original_pcs]

        feed_pc_v = np.array([x[0] for x in feed_pc_v])
        feed_pc_v_org = np.array([x[0] for x in feed_pc_v_org])

        ae_reconstructions, v_reconstructions, _ = ae.reconstruct([feed_pc, feed_pc_v], compute_loss=False)

        visualize(params['experiment_name'], params['class_name'], feed_pc, ae_reconstructions, feed_pc_v_org, v_reconstructions, 3)

        print('Finish visualize')


if __name__ == '__main__':
    main()
