def parse_config(argv=None):
    import configargparse
    arg_formatter = configargparse.ArgumentDefaultsHelpFormatter
    cfg_parser = configargparse.DefaultConfigFileParser
    description = 'articulated bps project'
    parser = configargparse.ArgParser(formatter_class=arg_formatter,
                                      config_file_parser_class=cfg_parser,
                                      description=description,
                                      prog='POP')

    # general settings                              
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument('--name', type=str, default='debug', help='name of a model/experiment. this name will be used for saving checkpoints and will also appear in saved examples')
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'resume', 'test', 'test_seen', 'test_unseen'], help='train/resume training/test, \
                                  for test_seen will evaluate on seen outfits unseen poses; test_unseen will test on unseen outfits (specified in configs/clo_config.yaml')

    # architecture related
    parser.add_argument('--hsize', type=int, default=256, help='hideen layer size of the ShapeDecoder mlp')
    parser.add_argument('--nf', type=int, default=32)
    parser.add_argument('--use_dropout', type=int, default=0, help='whether use dropout in the UNet')
    parser.add_argument('--up_mode', type=str, default='upconv',  choices=['upconv', 'upsample'], help='the method to upsample in the UNet')
    parser.add_argument('--latent_size', type=int, default=256, help='the size of a latent vector that conditions the unet, leave it untouched (it is there for historical reason)')
    parser.add_argument('--pix_feat_dim', type=int, default=64, help='dim of the pixel-wise latent code output by the UNet')
    parser.add_argument('--pos_encoding', type=int, default=0, help='use Positional Encoding (PE) for uv coords instead of plain concat')
    parser.add_argument('--posemb_incl_input', type=int, default=0, help='if use PE, then include original coords in the positional encoding')
    parser.add_argument('--num_emb_freqs', type=int, default=6, help='if use PE: number of frequencies used in the positional embedding')
    parser.add_argument('--c_geom', type=int, default=64, help='channels of the geometry feature map')
    parser.add_argument('--c_pose', type=int, default=64, help='dim of the pixel-wise latent code output by the pose Unet')
    parser.add_argument('--transf_scaling', type=float, default=0.02, help='scale the transformation matrix (empirically found: will slightly improve performance')
    parser.add_argument('--geom_layer_type', type=str, default='conv', choices=[None, 'unet', 'conv', 'gaussian', 'bottleneck'], help='type of the layers to process the geom featmap')
    parser.add_argument('--gaussian_kernel_size', type=int, default=5, help='size of the smoothing gaussian kernel if using a fixed gaussian filter for the geom_layer_type')

    # data related
    parser.add_argument('--dataset_type', type=str, default='cape', help='cape or resynth. for CAPE, will use SMPL in the code, for ReSynth will use SMPL-X.')
    parser.add_argument('--data_spacing', type=int, default=1, help='get every N examples from dataset (set N a large number for fast experimenting)')
    parser.add_argument('--query_posmap_size', type=int, default=256, help='size of the **query** UV positional map')
    parser.add_argument('--inp_posmap_size', type=int, default=128, help='size of UV positional **feature** map')
    parser.add_argument('--scan_npoints', type=int, default=-1, help='number of points used in the GT point set. By default -1 will use all points (40000);\
                                                                      setting it to another number N will randomly sample N points at each iteration as GT for training.')
    parser.add_argument('--dataset_subset_portion', type=float, default=1.0, help='the portion with which a subset from all training data is randomly chosen, value between 0 and 1 (for faster training)')
    parser.add_argument('--random_subsample_scan', type=int, default=0, help='wheter use the full dense scan point cloud as the GT for the optimization in the test-unseen-outfit scenario,\
                                                                              or randomly sample a subset of points from it at every optimization iteration')

    # loss func related
    parser.add_argument('--w_m2s', type=float, default=1e4, help='weight for the Chamfer loss part 1: (m)odel to (s)can, i.e. from the prediction to the GT points')
    parser.add_argument('--w_s2m', type=float, default=1e4, help='weight for the Chamfer loss part 2: (s)can to (m)odel, i.e. from the GT points to the predicted points')
    parser.add_argument('--w_normal', type=float, default=1.0, help='weight for the normal loss term')
    parser.add_argument('--w_rgl', type=float, default=2e3, help='weight for residual length regularization term')
    parser.add_argument('--w_latent_rgl', type=float, default=1.0, help='weight for regularization term on the geometric feature tensor')

    # training / eval related
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--decay_start', type=int, default=250, help='start to decay the regularization loss term from the X-th epoch')
    parser.add_argument('--rise_start', type=int, default=250, help='start to rise the normal loss term from the X-th epoch')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--decay_every', type=int, default=400, help='decaly the regularization loss weight every X epochs')
    parser.add_argument('--rise_every', type=int, default=400, help='rise the normal loss weight every X epochs')
    parser.add_argument('--val_every', type=int, default=20, help='validate every x epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--lr_geomfeat', type=float, default=5e-4, help='learning rate for the geometric feature tensor auto-decoding')
    parser.add_argument('--save_all_results', type=int, default=0, help='save the entire test set results at inference')
    parser.add_argument('--num_optim_iterations', type=int, default=401, help='number of optim iterations for the geom_featmap at test time')
    parser.add_argument('--num_unseen_frames', type=int, default=1, help='number of frames to be optimized at the test-unseen optimization process')

    args, _ = parser.parse_known_args()

    return args


def parse_outfits(exp_name):
    '''
    parse the seen/unseen outfits configuration defined in configs/clo_config.yaml
    '''
    import yaml

    clo_config_pth = 'configs/clo_config.yaml'
    config_all = yaml.load(open(clo_config_pth), Loader=yaml.SafeLoader) # config of all experiments

    config_exp = config_all[exp_name] # config of the specified experiment
    
    ret_dict = {}
    for key, value in config_exp.items():
        # value is a list of outfit names (e.g. 03375_shortlong)
        # now turn it into a dictionary where the key is the names and values are indexes (used for clothing label)
        value = sorted(value) # sort the clo type incase yaml loading doesn't preserve order which may case train/test time clo label discrepency on different machines
        value_dict = dict(zip(value, range(len(value))))
        ret_dict[key] = value_dict

    return ret_dict

