import os
from os.path import join, basename, dirname, realpath
import sys
import time
from datetime import date, datetime
import yaml

PROJECT_DIR = dirname(realpath(__file__))
LOGS_PATH = join(PROJECT_DIR, 'checkpoints')

SAMPLES_PATH = join(PROJECT_DIR, 'results', 'saved_samples')
sys.path.append(PROJECT_DIR)

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from lib.config_parser import parse_config, parse_outfits
from lib.dataset import CloDataSet
from lib.network import POP
from lib.train import train
from lib.infer import test_seen_clo, test_unseen_clo
from lib.utils_io import load_masks, load_barycentric_coords, save_model, save_latent_feats, load_latent_feats
from lib.utils_model import SampleSquarePoints
from lib.utils_train import adjust_loss_weights

torch.manual_seed(12345)
np.random.seed(12345)

DEVICE = torch.device('cuda')

def main():
    args = parse_config()

    exp_name = args.name
    
    # NOTE: when using your custom data, modify the following path to where the packed data is stored.
    data_root = join(PROJECT_DIR, 'data', '{}'.format(args.dataset_type.lower()), 'packed')

    log_dir = join(PROJECT_DIR,'tb_logs/{}/{}'.format(date.today().strftime('%m%d'), exp_name))
    ckpt_dir = join(LOGS_PATH, exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    samples_dir_val = join(SAMPLES_PATH, exp_name, 'val')
    samples_dir_test_seen_base = join(SAMPLES_PATH, exp_name, 'test_seen')
    samples_dir_test_unseen_base = join(SAMPLES_PATH, exp_name, 'test_unseen')
    os.makedirs(samples_dir_test_seen_base, exist_ok=True)
    os.makedirs(samples_dir_test_unseen_base, exist_ok=True)
    os.makedirs(samples_dir_val, exist_ok=True)

    body_model = 'smpl' if args.dataset_type.lower() == 'cape' else 'smplx'

    # uv locations, indices of the valid pixels and uv coordinates on the **query** (high-res) UV map
    flist_uv, valid_idx, uv_coord_map = load_masks(PROJECT_DIR, args.query_posmap_size, body_model=body_model)
    bary_coords = load_barycentric_coords(PROJECT_DIR, args.query_posmap_size, body_model=body_model)

    # parse names of the outfits used in the experiment
    outfits = parse_outfits(args.name)
    num_outfits_seen, num_outfits_unseen = len(outfits['seen']), len(outfits['unseen'])
    with open(join(LOGS_PATH, exp_name, 'clo_labels.yaml'), 'w') as fp:
        yaml.dump(outfits['seen'], fp, default_flow_style=False)

    # build_model
    model = POP(
                input_nc=3,
                c_pose=args.c_pose,
                c_geom=args.c_geom,
                inp_posmap_size=args.inp_posmap_size,
                hsize=args.hsize,
                nf=args.nf,
                up_mode=args.up_mode,
                use_dropout=bool(args.use_dropout),
                pos_encoding=bool(args.pos_encoding),
                num_emb_freqs=args.num_emb_freqs,
                posemb_incl_input=bool(args.posemb_incl_input),
                uv_feat_dim=2,
                geom_layer_type=args.geom_layer_type,
                gaussian_kernel_size=args.gaussian_kernel_size, 
                )
    # print(model)

    # geometric feature tensor
    geom_featmap = torch.ones(num_outfits_seen, args.c_geom, args.inp_posmap_size, args.inp_posmap_size).normal_(mean=0., std=0.01).cuda()
    geom_featmap.requires_grad = True

    # for compatibility with the SCALE code (https://github.com/qianlim/SCALE) we keep the 'patch sampling' notion, 
    # but only sample 1 point per 'patch'; i.e. there's no 'patch' any more, just points.
    subpixel_sampler = SampleSquarePoints(npoints=1)

    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters(), "lr": args.lr},
            {"params": geom_featmap, "lr": args.lr_geomfeat}
        ])

    n_epochs = args.epochs
    epoch_now = 0

    dataset_config = {
                     'dataset_type': args.dataset_type,
                     'body_model': body_model,
                     'data_root': data_root,
                     'query_posmap_size':args.query_posmap_size, 
                     'inp_posmap_size': args.inp_posmap_size,
                     }

    model_config = {
                    'device': DEVICE,
                    'flist_uv': flist_uv,
                    'valid_idx': valid_idx,
                    'uv_coord_map': uv_coord_map,
                    'bary_coords_map': bary_coords,
                    'transf_scaling': args.transf_scaling,
                    }
                    
    '''
    ------------ Load checkpoints in case of test or resume training ------------
    '''
    if args.mode.lower() in ['resume', 'test', 'test_seen', 'test_unseen']:
        checkpoints = sorted([fn for fn in os.listdir(ckpt_dir) if fn.endswith('_model.pt')])
        latest = join(ckpt_dir, checkpoints[-1])
        print('\n------------------------Loading checkpoint {}'.format(basename(latest)))
        ckpt_loaded = torch.load(latest)
        model.load_state_dict(ckpt_loaded['model_state'])

        checkpoints = sorted([fn for fn in os.listdir(ckpt_dir) if fn.endswith('_geom_featmap.pt')])
        checkpoint = join(ckpt_dir, checkpoints[-1])
        load_latent_feats(checkpoint, geom_featmap)

        if args.mode.lower() == 'resume':
            optimizer.load_state_dict(ckpt_loaded['optimizer_state'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(DEVICE)
            epoch_now = ckpt_loaded['epoch'] + 1
            print('\n------------------------Resume training from epoch {}'.format(epoch_now))

        if 'test' in args.mode.lower():
            epoch_idx = ckpt_loaded['epoch']
            model.to(DEVICE)
            print('\n------------------------Test model with checkpoint at epoch {}'.format(epoch_idx))


    '''
    ------------ Training from scratch, or resume from saved checkpoints ------------
    '''
    if args.mode.lower() in ['train', 'resume']:

        train_set = CloDataSet(split='train', outfits=outfits['seen'], sample_spacing=args.data_spacing,
                               dataset_subset_portion=args.dataset_subset_portion, **dataset_config)

        val_outfit_name, val_outfit_idx = list(outfits['seen'].items())[0]
        val_outfit = {val_outfit_name: val_outfit_idx}
        val_set = CloDataSet(split='test', outfits=val_outfit, sample_spacing=args.data_spacing, dataset_subset_portion=1.0, **dataset_config)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

        writer = SummaryWriter(log_dir=log_dir)

        print("Total: {} training examples, {} val examples. Training started..".format(len(train_set), len(val_set)))

        model.to(DEVICE)
        start = time.time()
        pbar = range(epoch_now, n_epochs)

        for epoch_idx in pbar:
            wdecay_rgl = adjust_loss_weights(args.w_rgl, epoch_idx, mode='decay', start=args.decay_start, every=args.decay_every)
            wrise_normal = adjust_loss_weights(args.w_normal, epoch_idx,  mode='rise', start=args.rise_start, every=args.rise_every)
            loss_weights = torch.tensor([args.w_s2m, args.w_m2s, wrise_normal, wdecay_rgl, args.w_latent_rgl])

            train_stats = train(model, geom_featmap, train_loader, optimizer,
                                loss_weights=loss_weights,
                                subpixel_sampler=subpixel_sampler,
                                **model_config)

            if epoch_idx % 50 == 0 or epoch_idx == n_epochs - 1:
                ckpt_path = join(ckpt_dir, '{}_epoch{}_model.pt'.format(exp_name, str(epoch_idx).zfill(5)))
                save_model(ckpt_path, model, epoch_idx, optimizer=optimizer)
                ckpt_path = join(ckpt_dir, '{}_epoch{}_geom_featmap.pt'.format(exp_name, str(epoch_idx).zfill(5)))
                save_latent_feats(ckpt_path, geom_featmap, epoch_idx)

            # test on val set every N epochs
            if epoch_idx % args.val_every == 0:
                dur = (time.time() - start) / (60 * (epoch_idx-epoch_now+1))
                now = datetime.now()
                dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                print('\n{}, Epoch {}, average {:.2f} min / epoch.'.format(dt_string, epoch_idx, dur))
                print('Weights s2m: {:.1e}, m2s: {:.1e}, normal: {:.1e}, rgl: {:.1e}'.format(args.w_s2m, args.w_m2s, wrise_normal, wdecay_rgl))

                # validate on the validation set of seen clothing, unsee poses
                val_stats = test_seen_clo(  
                                            model, 
                                            geom_featmap,
                                            val_loader, epoch_idx,
                                            samples_dir_val,
                                            subpixel_sampler=subpixel_sampler,
                                            model_name=exp_name,
                                            save_all_results=bool(args.save_all_results),
                                            mode='val',
                                            **model_config
                                        )

                val_total_loss = np.stack(val_stats).dot(loss_weights)
                val_stats.append(np.array(val_total_loss))

                tensorboard_tabs = ['model2scan', 'scan2model', 'normal_loss', 'residual_square', 'latent_rgl', 'total_loss']
                stats = {'train': train_stats, 'val': val_stats}

                for split in ['train', 'val']:
                    for (tab, stat) in zip(tensorboard_tabs, stats[split]):
                        writer.add_scalar('{}/{}'.format(tab, split), stat, epoch_idx)


        end = time.time()
        t_total = (end - start) / 60
        print("Training finished, duration: {:.2f} minutes. Now eval on test set..\n".format(t_total))
        writer.close()


    '''
    ------------ Test model, seen outfits ------------
    '''
    if args.mode.lower() in ['train', 'test', 'test_seen']:

        test_rst_msg = []
        test_rst_msg.append('\n\n{}, epoch={}, test query resolution={} \n'.format(exp_name, epoch_idx, args.query_posmap_size))

        print('\n------------------------Eval on test data, seen outfits, unseen poses...')

        per_outfit_dataset = [{k:v} for k, v in outfits['seen'].items()]

        sum_chamfer_all_outfits, sum_normal_all_outfts, num_ex_all_outfits = 0, 0, 0

        test_rst_msg.append('\tEval on test set, seen clo:\n')

        for outfit in per_outfit_dataset: # outfit is a dict that contains a single key:val pair (a clothing type)

            test_set = CloDataSet(split='test', outfits=outfit, sample_spacing=args.data_spacing, dataset_subset_portion=1.0, **dataset_config)
            test_loader = DataLoader(test_set, batch_size=args.batch_size*2, shuffle=False, num_workers=4)

            samples_dir_outfit = join(samples_dir_test_seen_base, 'query_resolution{}'.format(args.query_posmap_size), list(outfit.keys())[0])
            os.makedirs(samples_dir_outfit, exist_ok=True)
            
            start = time.time()
            test_stats = test_seen_clo( 
                                        model, geom_featmap, test_loader, epoch_idx,
                                        samples_dir_outfit,
                                        mode='test_seen',
                                        subpixel_sampler=subpixel_sampler,
                                        model_name=exp_name,
                                        save_all_results=bool(args.save_all_results),
                                        **model_config
                                    )
            test_m2s, test_s2m, test_lnormal, _, _ = test_stats

            # accumulate errors across all outfits
            sum_chamfer_outfit = (test_m2s+test_s2m) * len(test_set) 
            sum_normal_outfit = test_lnormal * len(test_set)

            sum_chamfer_all_outfits += sum_chamfer_outfit
            sum_normal_all_outfts += sum_normal_outfit
            num_ex_all_outfits += len(test_set)

            outfit_info = '{:<18}, {} examples.'.format(list(outfit.keys())[0], len(test_set))
            test_seen_result = "{:<34} m2s dist: {:.3e}, s2m dist: {:.3e}. Chamfer total: {:.3e}, normal loss: {:.3e}.\n"\
                            .format(outfit_info, test_m2s, test_s2m, test_m2s+test_s2m, test_lnormal)
            print(test_seen_result)
            test_rst_msg.append('\t\t{}'.format(test_seen_result))
        
        # calculate the average error across all outfits
        avg_chamfer_all = sum_chamfer_all_outfits / num_ex_all_outfits
        avg_normal_all = sum_normal_all_outfts / num_ex_all_outfits
        test_seen_full_stats = '\t\tOn all seen data, {} exmaples, average Chamfer: {:.3e}, average normal loss: {:.3e}\n'\
            .format(num_ex_all_outfits, avg_chamfer_all, avg_normal_all)
        test_rst_msg.append(test_seen_full_stats)


    '''
    ------------ Test model, unseen outfits ------------
    '''
    if args.mode.lower() in ['test', 'test_unseen']:
        test_rst_msg = []
        test_rst_msg.append('\n\n{}, epoch={}, test query resolution={} \n'.format(exp_name, epoch_idx, args.query_posmap_size))

        print('\n------------------------Eval on test data, unseen outfit, unseen poses...')

        per_outfit_dataset = [{k:v} for k, v in outfits['unseen'].items()]

        test_rst_msg.append('\tEval on test set, unseen clo:')

        for outfit in per_outfit_dataset:
            assert args.num_unseen_frames ==1, "Currently only supports single scan optimization."
            
            print('------Sequence test data for animation:')
            test_set = CloDataSet(split='test', outfits=outfit, sample_spacing=args.data_spacing, dataset_subset_portion=1.0, **dataset_config)
            test_loader = DataLoader(test_set, batch_size=args.batch_size*2, shuffle=False, num_workers=4)

            print('------Single frame scan data for optimization:')
            data_spacing_for_optim = len(test_set) // args.num_unseen_frames
            test_set_for_optim = CloDataSet(split='test', outfits=outfit, sample_spacing=data_spacing_for_optim, dataset_subset_portion=1.0, **dataset_config)
            test_loader_for_optim = DataLoader(test_set_for_optim, batch_size=args.batch_size, shuffle=False, num_workers=4)

            samples_dir_outfit = join(samples_dir_test_unseen_base, 'query_resolution{}'.format(args.query_posmap_size))
            
            # loss weights for the optimization w.r.t. the unseen scan
            wdecay_rgl = adjust_loss_weights(args.w_rgl, epoch_idx, mode='decay', start=args.decay_start, every=args.decay_every)
            wrise_normal = adjust_loss_weights(args.w_normal, epoch_idx,  mode='rise', start=args.rise_start, every=args.rise_every)
            loss_weights = torch.tensor([args.w_s2m, args.w_m2s, wrise_normal, wdecay_rgl, args.w_latent_rgl])
        
            test_stats = test_unseen_clo(
                                        model,
                                        geom_featmap,
                                        test_loader, 
                                        test_loader_for_optim, 
                                        epoch_idx,
                                        samples_dir_outfit,
                                        mode='test_unseen',
                                        model_name=exp_name,
                                        subpixel_sampler=subpixel_sampler,
                                        loss_weights=loss_weights,
                                        dataset_type=args.dataset_type,
                                        num_optim_iterations=args.num_optim_iterations,
                                        random_subsample_scan=bool(args.random_subsample_scan),
                                        save_all_results=bool(args.save_all_results),
                                        **model_config
                                        )

            test_m2s, test_s2m, test_lnormal, _, _ = test_stats

            outfit_info = '{:<18}, {} examples.'.format(list(outfit.keys())[0], len(test_set))
            test_unseen_result = "{:<34} m2s dist: {:.3e}, s2m dist: {:.3e}. Chamfer total: {:.3e}, normal loss: {:.3e}.\n"\
                                            .format(outfit_info, test_m2s, test_s2m, test_m2s+test_s2m, test_lnormal)
            print(test_unseen_result)
            test_rst_msg.append('\t\t{}'.format(test_unseen_result))


if __name__ == '__main__':
    main()