import os.path as osp

from src.ae_templates import mlp_architecture_tl_net, default_train_params
from src.autoencoder import Configuration as Conf
from src.point_net_ae import PointNetAutoEncoder

from src.in_out import snc_category_to_synth_id, create_dir, PointCloudDataSet, \
                                        load_all_point_clouds_under_folder

from src.tf_utils import reset_tf_graph


def get_conf(class_name):
    top_out_dir = './data/'          # Use to save Neural-Net check-points etc.
    experiment_name = '{}_tl'.format(class_name)
    n_pc_points = 2048                # Number of points per model.
    bneck_size = 128                  # Bottleneck-AE size
    ae_loss = 'emd'                   # Loss to optimize: 'emd' or 'chamfer'

    train_params = default_train_params()

    encoder, decoder, embedder, enc_args, dec_args, emb_args = mlp_architecture_tl_net(n_pc_points, bneck_size)
    train_dir = create_dir(osp.join(top_out_dir, experiment_name))

    conf = Conf(n_input = [n_pc_points, 3],
            loss = ae_loss,
            training_epochs = train_params['training_epochs'],
            batch_size = train_params['batch_size'],
            denoising = train_params['denoising'],
            learning_rate = train_params['learning_rate'],
            train_dir = train_dir,
            loss_display_step = train_params['loss_display_step'],
            saver_step = train_params['saver_step'],
            z_rotate = train_params['z_rotate'],
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

    all_pc_data = load_all_point_clouds_under_folder(train_dir, class_name, n_threads=20, file_ending='.ply', verbose=True)
    train_stats = ae.train(all_pc_data, conf, log_file=fout)
    fout.close()

    feed_pc, feed_pc_v, feed_model_names, _ = all_pc_data.next_batch(10)
    ae_reconstructions, v_reconstructions, _ = ae.reconstruct([feed_pc, feed_pc_v])
    # latent_codes = ae.transform(feed_pc)

if __name__ == '__main__':
    main()
