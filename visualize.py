import os
import pickle
import argparse
import numpy as np
import matplotlib

from src.point_net_ae import PointNetAutoEncoder
from src.in_out import PointCloudDataSet, load_all_point_clouds_under_folder
from src.tf_utils import reset_tf_graph
from src.general_utils import get_conf, get_visible_points


import plotly.offline as offline
import plotly.graph_objs as go


def parse_args():
    params = {'class_name': None,
          'n_pc_points': 2048,
          'batch_size': 50,
          'training_epochs': 2000,
          'denoising': False,
          'gauss_augment': False,
          'learning_rate': 0.0005,
          'z_rotate': True,
          'saver_step': 100,
          'input_color':True,
          'output_color': False,
          'loss_display_step': 1,
          'experiment_name': 'tmp'
          }

    parser = argparse.ArgumentParser(
            prog="Visualizing PC2PC TL network",
            usage="python visualize.py [class_name] [options]", #Usage
            add_help = True
            )

    parser.add_argument("class_name", help="Name of class (e.g. 'chair' or 'all' or 'all_w_clutter')")
    parser.add_argument("experiment_name", help="Name of experiment")
    parser.add_argument("-n", "--n_points", help="Number of input points", default=2048, type=int)
    parser.add_argument("-c", "--input_color", action='store_true', help="Add color to input feature")
    parser.add_argument("-a", "--actual_name", help="Name of Experiment currently saved in ./data/ (to be deleted)", type=str)

    args = parser.parse_args()

    params['class_name'] = args.class_name
    params['experiment_name'] = args.experiment_name
    params['n_pc_points'] = args.n_points
    params['input_color'] = args.input_color
    params['actual_name'] = args.actual_name

    return params



def get_trace(_x, _y, _z, _color='rgb(255, 0, 0)'):
    trace = go.Scatter3d(
        x=_x,
        y=_y,
        z=_z,
        mode='markers',
        marker=dict(
            sizemode='diameter',
            color = _color,
            colorscale = 'Portland',
            line=dict(color='rgb(255, 255, 255)'),
            opacity=0.9,
            size=3
        )
    )
    return trace



def visualize(experiment_name, class_name, n=0):
    gt_file = os.path.join('./result', '_'.join([experiment_name, class_name, 'gt.pkl']))
    rec_file = os.path.join('./result', '_'.join([experiment_name, class_name, 'rec.pkl']))
    vgt_file = os.path.join('./result', '_'.join([experiment_name, class_name, 'v_gt.pkl']))
    vrec_file = os.path.join('./result', '_'.join([experiment_name, class_name, 'v_rec.pkl']))

    gt = np.load(gt_file)[n]
    rec = np.load(rec_file)[n]
    vgt = np.load(vgt_file)[n]
    vrec = np.load(vrec_file)[n]

    trace1 = get_trace(rec[:, 0], rec[:, 1], rec[:, 2], 'rgb(255, 0, 0)') # reconstructed by AE, red
    trace2 = get_trace(gt[:, 0], gt[:, 1], gt[:, 2], 'rgb(0, 0, 255)') # GT (input), blue
    layout=dict(height=600, width=600, title='_'.join([experiment_name, class_name, 'AE']))
    fig=dict(data=[trace1, trace2], layout=layout)
    offline.plot(fig, filename=os.path.join('./html', '_'.join([experiment_name, class_name, 'AE'])))


    trace1 = get_trace(vrec[:, 0], vrec[:, 1], vrec[:, 2], 'rgb(255, 0, 0)') # reconstructed by TL, red
    trace2 = get_trace(vgt[:, 0], vgt[:, 1], vgt[:, 2], 'rgb(0, 0, 255)') # input (visible surface), blue
    layout=dict(height=600, width=600, title='_'.join([experiment_name, class_name, 'TL']))
    fig=dict(data=[trace1, trace2], layout=layout)
    offline.plot(fig, filename=os.path.join('./html', '_'.join([experiment_name, class_name, 'TL'])))




def main():
    params = parse_args()
    conf = get_conf(params)

    ckpt_path = os.path.join('./data/', params['experiment_name'], params['class_name'])

    epoch = 2000

    test_dir = './s3dis/Area_6/*/Annotations/{}_*.txt'

    reset_tf_graph()
    ae = PointNetAutoEncoder(conf.experiment_name, conf)
    ae.restore_model(ckpt_path, epoch, verbose=True)

    all_pc_data = load_all_point_clouds_under_folder(test_dir, params['class_name'], n_points=conf.n_input[0], with_color=conf.input_color, n_threads=20, verbose=True)

    feed_pc, feed_model_names, _ = all_pc_data.next_batch(10)
    feed_pc_v, feed_pc_v_org = np.split(np.array([get_visible_points(x, org=True) for x in feed_pc]), 2, axis=1) # [normalized_pcs, original_pcs]

    feed_pc_v = np.array([x[0] for x in feed_pc_v])
    feed_pc_v_org = np.array([x[0] for x in feed_pc_v_org])

    ae_reconstructions, v_reconstructions, _ = ae.reconstruct([feed_pc, feed_pc_v], compute_loss=False)

    with open('result/{}_{}_gt.pkl'.format(params['experiment_name'], params['class_name']), mode='wb') as f: # actual_name will be changed to experiment_name
        pickle.dump(feed_pc, f)
    with open('result/{}_{}_v_gt.pkl'.format(params['experiment_name'], params['class_name']), mode='wb') as f:
        pickle.dump(feed_pc_v_org, f)
    with open('result/{}_{}_rec.pkl'.format(params['experiment_name'], params['class_name']), mode='wb') as f:
        pickle.dump(ae_reconstructions, f)
    with open('result/{}_{}_v_rec.pkl'.format(params['experiment_name'], params['class_name']), mode='wb') as f:
        pickle.dump(v_reconstructions, f)
    print('finish inference')

    n = 0
    visualize(params['experiment_name'], params['class_name'], n)

if __name__ == '__main__':
    main()
