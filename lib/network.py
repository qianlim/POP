import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils_model import PositionalEncoding, normalize_uv, uv_to_grid
from lib.modules import UnetNoCond5DS, UnetNoCond6DS, UnetNoCond7DS, GeomConvLayers, GaussianSmoothingLayers, GeomConvBottleneckLayers, ShapeDecoder


class POP(nn.Module):
    def __init__(
                self, 
                input_nc=3, # num channels of the unet input
                c_geom=64, # channels of the geometric features
                c_pose=64, # channels of the pose features
                geom_layer_type='conv', # the type of architecture used for smoothing the geometric feature tensor
                nf=64, # num filters for the unet
                inp_posmap_size=128, # size of UV positional map (pose conditioning), i.e. the input to the pose unet
                hsize=256, # hidden layer size of the ShapeDecoder MLP
                up_mode='upconv', # upconv or upsample for the upsampling layers in the pose feature UNet
                use_dropout=False, # whether use dropout in the pose feature UNet
                pos_encoding=False, # whether use Positional Encoding for the query UV coordinates
                num_emb_freqs=8, # number of sinusoida frequences if positional encoding is used
                posemb_incl_input=False, # wheter include the original coordinate if using Positional Encoding
                uv_feat_dim=2, # input dimension of the uv coordinates
                pq_feat_dim = 2, # input dimension of the pq coordinates
                gaussian_kernel_size = 3, #optional, if use gaussian smoothing for the geometric features
                ):

        super().__init__()
        self.inp_posmap_size = inp_posmap_size
        self.pos_encoding = pos_encoding
        self.geom_layer_type = geom_layer_type 
        
        geom_proc_layers = {
            'unet': UnetNoCond5DS(c_geom, c_geom, nf, up_mode, use_dropout), # use a unet
            'conv': GeomConvLayers(c_geom, c_geom, c_geom, use_relu=False), # use 3 trainable conv layers
            'bottleneck': GeomConvBottleneckLayers(c_geom, c_geom, c_geom, use_relu=False), # use 3 trainable conv layers
            'gaussian': GaussianSmoothingLayers(channels=c_geom, kernel_size=gaussian_kernel_size, sigma=1.0), # use a fixed gaussian smoother
        }
        unets = {32: UnetNoCond5DS, 64: UnetNoCond6DS, 128: UnetNoCond7DS, 256: UnetNoCond7DS}
        unet_loaded = unets[self.inp_posmap_size]

        if self.pos_encoding:
            self.embedder = PositionalEncoding(num_freqs=num_emb_freqs,
                                               input_dims=uv_feat_dim,
                                               include_input=posemb_incl_input)
            self.embedder.create_embedding_fn()
            self.uv_feat_dim = self.embedder.out_dim

        # U-net: for extracting pixel-aligned pose features from the input UV positional maps
        self.unet_posefeat = unet_loaded(input_nc, c_pose, nf, up_mode=up_mode, use_dropout=use_dropout)

        # optional layer for spatially smoothing the geometric feature tensor
        if geom_layer_type is not None:
            self.geom_proc_layers = geom_proc_layers[geom_layer_type]
    
        # shared shape decoder across different outfit types
        self.decoder = ShapeDecoder(in_size=uv_feat_dim + pq_feat_dim + c_geom + c_pose,
                                    hsize=hsize, actv_fn='softplus')

            
    def forward(self, x, geom_featmap, uv_loc, pq_coords):
        '''
        :param x: input posmap, [batch, 3, 256, 256]
        :param geom_featmap: a [B, C, H, W] tensor, spatially pixel-aligned with the pose features extracted by the UNet
        :param uv_loc: querying uv coordinates, ranging between 0 and 1, of shape [B, H*W, 2].
        :param pq_coords: the 'sub-UV-pixel' (p,q) coordinates, range [0,1), shape [B, H*W, 1, 2]. 
                        Note: It is the intra-patch coordinates in SCALE. Kept here for the backward compatibility with SCALE.
        :return:
            clothing offset vectors (residuals) and normals of the points
        '''
        # pose features
        pose_featmap = self.unet_posefeat(x)

        # geometric feature tensor
        if self.geom_layer_type is not None:
            geom_featmap = self.geom_proc_layers(geom_featmap)

        # pose and geom features are concatenated to form the feature for each point
        pix_feature = torch.cat([pose_featmap, geom_featmap], 1)

        feat_res = geom_featmap.shape[2] # spatial resolution of the input pose and geometric features
        uv_res = int(uv_loc.shape[1]**0.5) # spatial resolution of the query uv map

        # spatially bilinearly upsample the features to match the query resolution
        if feat_res != uv_res:
            query_grid = uv_to_grid(uv_loc, uv_res)
            pix_feature = F.grid_sample(pix_feature, query_grid, mode='bilinear', align_corners=False)#, align_corners=True)

        B, C, H, W = pix_feature.shape
        N_subsample = 1 # inherit the SCALE code custom, but now only sample one point per pixel

        uv_feat_dim = uv_loc.size()[-1]
        pq_coords = pq_coords.reshape(B, -1, 2).transpose(1, 2)  # [B, 2, Num all subpixels]
        uv_loc = uv_loc.expand(N_subsample, -1, -1, -1).permute([1, 2, 0, 3])

        # uv and pix feature is shared for all points in each patch
        pix_feature = pix_feature.view(B, C, -1).expand(N_subsample, -1,-1,-1).permute([1,2,3,0]) # [B, C, N_pix, N_sample_perpix]
        pix_feature = pix_feature.reshape(B, C, -1)

        if self.pos_encoding:
            uv_loc = normalize_uv(uv_loc).view(-1,uv_feat_dim)
            uv_loc = self.embedder.embed(uv_loc).view(B, -1,self.embedder.out_dim).transpose(1,2)
        else:
            uv_loc = uv_loc.reshape(B, -1, uv_feat_dim).transpose(1, 2)  # [B, N_pix, N_subsample, 2] --> [B, 2, Num of all pq subpixels]

        residuals, normals = self.decoder(torch.cat([pix_feature, uv_loc, pq_coords], 1))  # [B, 3, N all subpixels]

        residuals = residuals.view(B, 3, H, W, N_subsample)
        normals = normals.view(B, 3, H, W, N_subsample)
        
        return residuals, normals
