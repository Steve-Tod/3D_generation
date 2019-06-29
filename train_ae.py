import os
import argparse, json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='params for ae training')
    parser.add_argument('-o', '--opt', type=str, required=True)
    args = parser.parse_args()

    with open(args.opt, 'r') as f:
        opt = json.load(f)

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
        [str(x) for x in opt['gpu_ids']])

    import os.path as osp
    from data.pointcloud_dataset import load_one_class_under_folder
    from trainers.autoencoder import default_train_params, Configuration
    from utils.dirs import mkdir_and_rename
    from utils.tf import reset_tf_graph

    if opt['model']['type'] == 'ae':
        from models.ae_models import mlp_architecture_ala_iclr_18 as ae_arch
        from trainers.point_net_ae import PointNetAutoEncoder as PAE
    elif opt['model']['type'] == 'vae':
        from models.vae_models import mlp_architecture_ala_iclr_18 as ae_arch
        from trainers.point_net_vae import PointNetVariationalAutoEncoder as PAE
    else:
        raise NotImplementedError('model type %s not implemented!' %
                                  opt['model']['type'])

    # dataset
    pc_dataset = load_one_class_under_folder(opt['data']['data_root'],
                                             opt['data']['class_name'],
                                             verbose=True)

    # model
    encoder, decoder, enc_args, dec_args = ae_arch(opt['model']['num_points'],
                                                   opt['model']['bneck_size'])

    # path and trainer
    train_dir = osp.join(opt['path']['train_root'],
                         opt['path']['experiment_name'])
    train_params = opt['train']
    if train_params['resume']:  # restore
        conf = Configuration.load(train_dir + '/configuration')
        reset_tf_graph()
        ae = PAE(conf.experiment_name, conf)
        ae.restore_model(conf.save_dir, epoch=train_params['restore_epoch'])
    else:
        mkdir_and_rename(osp.join(train_dir, 'checkpoint'))
        with open(osp.join(train_dir, 'opt.json'), 'w') as f:
            json.dump(opt, f, indent=4)
        conf = Configuration(
            n_input=[opt['model']['num_points'], 3],
            loss=opt['train']['ae_loss'],
            training_epochs=train_params['training_epochs'],
            batch_size=train_params['batch_size'],
            denoising=train_params['denoising'],
            learning_rate=train_params['learning_rate'],
            train_dir=train_dir,
            loss_display_step=train_params['loss_display_step'],
            saver_step=train_params['saver_step'],
            z_rotate=train_params['z_rotate'],
            encoder=encoder,
            decoder=decoder,
            encoder_args=enc_args,
            decoder_args=dec_args)
        # latent weight
        if 'latent_vs_recon' in train_params.keys():
            import numpy as np
            conf.latent_vs_recon = np.array([train_params['latent_vs_recon']],
                                            dtype=np.float32)[0]
        conf.experiment_name = opt['path']['experiment_name']
        conf.held_out_step = 5  # How often to evaluate/print out loss on
        # held_out data (if they are provided in ae.train() ).
        conf.save(osp.join(train_dir, 'configuration'))
        reset_tf_graph()
        ae = PAE(conf.experiment_name, conf)

    print(conf)
    buf_size = 1  # Make 'training_stats' file to flush each output line regarding training.
    fout = open(osp.join(conf.train_dir, 'train_stats.txt'), 'a', buf_size)
    train_stats = ae.train(pc_dataset, conf, log_file=fout)
    fout.close()
