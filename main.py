import os.path as osp

from src.point_net_ae import PointNetAutoEncoder
from src.in_out import PointCloudDataSet, load_all_point_clouds_under_folder
from src.tf_utils import reset_tf_graph
from src.general_utils import get_conf


def main():
    class_name = 'chair' #raw_input('Give me the class name (e.g. "chair"): ').lower()
    class_name = raw_input('Give me the class name (e.g. "chair" or "all" or "all_w_clutter"): ').lower()

    conf = get_conf(class_name)

    train_dir = './s3dis/Area_[1-5]/*/Annotations/{}_*.txt'
    # train_dir = './s3dis/Area_4/*/Annotations/{}_*.txt'
    test_dir = './s3dis/Area_6/*/Annotations/{}_*.txt'

    reset_tf_graph()
    ae = PointNetAutoEncoder(conf.experiment_name, conf)

    buf_size = 1 # Make 'training_stats' file to flush each output line regarding training.
    fout = open(osp.join(conf.train_dir, 'train_stats.txt'), 'a', buf_size)

    all_pc_data = load_all_point_clouds_under_folder(train_dir, class_name, with_color=conf.input_color, n_threads=20, verbose=True)
    train_stats = ae.train(all_pc_data, conf, log_file=fout)
    fout.close()

    feed_pc, feed_pc_v, feed_model_names, _ = all_pc_data.next_batch(10)
    ae_reconstructions, v_reconstructions, _ = ae.reconstruct([feed_pc, feed_pc_v])
    # latent_codes = ae.transform(feed_pc)


if __name__ == '__main__':
    main()
