import os
from os.path import join, dirname
import glob

import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

class CloDataSet(Dataset):
    def __init__(self, data_root=None, split='train', body_model='smpl', dataset_type='cape',
                 sample_spacing=1, query_posmap_size=256, inp_posmap_size=128, scan_npoints=-1, 
                 dataset_subset_portion=1.0, outfits={}):

        self.dataset_type = dataset_type
        self.data_root = data_root
        self.data_dirs = {outfit: join(data_root, outfit, split) for outfit in outfits.keys()} # will be sth like "./data/packed/cape/00032_shortlong/train"
        self.dataset_subset_portion = dataset_subset_portion # randomly subsample a number of data from each clothing type (using all data from all outfits will be too much)
        self.query_posmap_size = query_posmap_size
        self.inp_posmap_size = inp_posmap_size

        self.split = split
        self.query_posmap_size = query_posmap_size
        self.spacing = sample_spacing
        self.scan_npoints = scan_npoints
        self.f = np.load(join(SCRIPT_DIR, '..', 'assets', '{}_faces.npy'.format(body_model)))
        self.clo_label_def = outfits


        self.posmap, self.posmap_meanshape, self.scan_n, self.scan_pc = [], [], [], []
        self.scan_name, self.body_verts, self.clo_labels =  [], [], []
        self.vtransf = []
        self._init_dataset()
        self.data_size = int(len(self.posmap))

        print('Data loaded, in total {} {} examples.\n'.format(self.data_size, self.split))

    def _init_dataset(self):
        print('Loading {} data...'.format(self.split))

        flist_all = []
        subj_id_all = []

        for outfit_id, (outfit, outfit_datadir) in enumerate(self.data_dirs.items()):
            flist = sorted(glob.glob(join(outfit_datadir, '*.npz')))[::self.spacing]
            print('Loading {}, {} examples..'.format(outfit, len(flist)))
            flist_all = flist_all + flist
            subj_id_all = subj_id_all + [outfit.split('_')[0]] * len(flist)

        if self.dataset_subset_portion < 1.0:
            import random
            random.shuffle(flist_all)
            num_total = len(flist_all)
            num_chosen = int(self.dataset_subset_portion*num_total)
            flist_all = flist_all[:num_chosen]
            print('Total examples: {}, now only randomly sample {} from them...'.format(num_total, num_chosen))

        for idx, fn in enumerate(tqdm(flist_all)):
            dd = np.load(fn)
            clo_type = dirname(fn).split('/')[-2] # e.g. longlong 
            clo_label = self.clo_label_def[clo_type] # the numerical label of the type in the lookup table (outfit_labels.json)
            self.clo_labels.append(torch.tensor(clo_label).long())
            self.posmap.append(torch.tensor(dd['posmap{}'.format(self.query_posmap_size)]).float().permute([2,0,1]))

            # for historical reasons in the packed data the key is called "posmap_canonical"
            # it actually stands for the positional map of the *posed, mean body shape* of SMPL/SMPLX (see POP paper Sec 3.2)
            # which corresponds to the inp_posmap_ms in the train and inference code 
            # if the key is not available, simply use each subject's personalized body shape.
            if 'posmap_canonical{}'.format(self.inp_posmap_size) not in dd.files:
                self.posmap_meanshape.append(torch.tensor(dd['posmap{}'.format(self.inp_posmap_size)]).float().permute([2,0,1]))
            else:
                self.posmap_meanshape.append(torch.tensor(dd['posmap_canonical{}'.format(self.inp_posmap_size)]).float().permute([2,0,1]))
            self.scan_n.append(torch.tensor(dd['scan_n']).float())
            
            # in the packed files the 'scan_name' field doensn't contain subj id, need to append it
            scan_name_loaded = str(dd['scan_name'])
            scan_name = scan_name_loaded if scan_name_loaded.startswith('0') else '{}_{}'.format(subj_id_all[idx], scan_name_loaded)
            self.scan_name.append(scan_name)

            self.body_verts.append(torch.tensor(dd['body_verts']).float())
            self.scan_pc.append(torch.tensor(dd['scan_pc']).float())
            
            vtransf = torch.tensor(dd['vtransf']).float()
            if vtransf.shape[-1] == 4:
                vtransf = vtransf[:, :3, :3]
            self.vtransf.append(vtransf)


    def __getitem__(self, index):
        posmap = self.posmap[index]
        posmap_meanshape = self.posmap_meanshape[index] # in mean SMPL/ SMPLX body shape but in the same pose as the original subject
        scan_name = self.scan_name[index]
        body_verts = self.body_verts[index]
        clo_label = self.clo_labels[index]

        scan_n = self.scan_n[index]
        scan_pc = self.scan_pc[index]

        vtransf = self.vtransf[index]

        if self.scan_npoints != -1: 
            selected_idx = torch.randperm(len(scan_n))[:self.scan_npoints]
            scan_pc = scan_pc[selected_idx, :]
            scan_n = scan_n[selected_idx, :]

        return posmap, posmap_meanshape, scan_n, scan_pc, vtransf, scan_name, body_verts, clo_label

    def __len__(self):
        return self.data_size