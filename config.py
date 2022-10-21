config = {      'multi-gpus': True,
           'enable_autocast': True,
                   "dataset": 'PPHM', 
               'dataset_dir': './datasets/PPHM',
                      'mode': 'train',
               'results_dir': './results',
                    'g_loss': 'loss_vanilla_gen',
                    'd_loss': 'loss_vanilla_simplegp_dis', 
               'SN'         : False,   #spectral normalization
               'OR'         : False,  #orthogonal regularization
               'num_workers': 0,     # num_workers for Dataloader
               'random_flip': True, 
             'random_rotate': 0.0,
                    'resize': False,
                'image_size': 512,
               'load_in_mem': False,
               'inplace_act': True,
             'resblock_bias': True, 
                'c_dim'     : 4,
                'conv_dim'  : 64,
              'downsample_g': 2,
                'repeat_num': 6,
              'downsample_d': 6,
                'patchGAN'  : False,
                 'start_itr': 0,
                'start_kimg': 0, 
               'start_epoch': 0,
                'sample_itr': 20000, 
                'log_itr'   :  -1,  # log singular values
              'snapshot_itr': 20000,
                       'lr' : 1e-5,
                    'cos_lr': True,
                 'clip_norm': 10.0,
                 'n_critics': 3, 
                'lambda_cls': 1.0, 
              'lambda_recon': 1.0,
                 'minibatch': 64,  # total batch size,  64 for A100*8 with autocast
                 'seed'     :  1234, 
                 'restart'  :   0, # restart kimg
                 'cos_kimg' : 10000,
                 'stop_kimg': 20000, 
         }


 # above config is for Model 3. comment out to train Model 1 and 2 in the paper
 """
config.update({
                    'g_loss': 'loss_hinge_gen',         
                    'd_loss': 'loss_hinge_dis',
                      'SN'  : True,   #spectral normalization
                      'OR'  : True,  #orthogonal regularization
                  'log_itr' :  500,  # log singular values
                   'seed'   :  1234, # change seed to train with different random seed
 })
"""