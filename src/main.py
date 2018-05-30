import os.path as osp

from ae_templates import mlp_architecture_ala_iclr_18, default_train_params
from autoencoder import Configuration as Conf
from point_net_ae import PointNetAutoEncoder

from in_out import snc_category_to_synth_id, create_dir, PointCloudDataSet, \
                                        load_all_point_clouds_under_folder

from tf_utils import reset_tf_graph


def get_conf(class_name):
    top_out_dir = '../data/'          # Use to save Neural-Net check-points etc.
    top_in_dir = '../data/shape_net_core_uniform_samples_2048/' # Top-dir of where point-clouds are stored.

    experiment_name = 'single_class_ae'
    n_pc_points = 2048                # Number of points per model.
    bneck_size = 128                  # Bottleneck-AE size
    ae_loss = 'emd'                   # Loss to optimize: 'emd' or 'chamfer'

    syn_id = snc_category_to_synth_id()[class_name]
    class_dir = osp.join(top_in_dir , syn_id)
    all_pc_data = load_all_point_clouds_under_folder(class_dir, n_threads=8, file_ending='.ply', verbose=True)

    train_params = default_train_params()

    encoder, decoder, enc_args, dec_args = mlp_architecture_ala_iclr_18(n_pc_points, bneck_size)
    train_dir = create_dir(osp.join(top_out_dir, experiment_name))

    print(enc_args)
    print(dec_args)

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
            encoder_args = enc_args,
            decoder_args = dec_args
           )
    conf.experiment_name = experiment_name
    conf.held_out_step = 5   # How often to evaluate/print out loss on 
                             # held_out data (if they are provided in ae.train() ).
    conf.save(osp.join(train_dir, 'configuration'))

    return conf



def main():
    class_name = raw_input('Give me the class name (e.g. "chair"): ').lower()
    conf = get_conf(class_name)

    reset_tf_graph()
    ae = PointNetAutoEncoder(conf.experiment_name, conf)

    buf_size = 1 # Make 'training_stats' file to flush each output line regarding training.
    fout = open(osp.join(conf.train_dir, 'train_stats.txt'), 'a', buf_size)
    train_stats = ae.train(all_pc_data, conf, log_file=fout)
    fout.close()

    feed_pc, feed_model_names, _ = all_pc_data.next_batch(10)
    reconstructions = ae.reconstruct(feed_pc)
    latent_codes = ae.transform(feed_pc)



if __name__ == '__main__':
    main()