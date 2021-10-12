import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def gen_transf_mtx_full_uv(verts, faces):
    '''
    given a positional uv map, for each of its pixel, get the matrix that transforms the prediction from local to global coordinates
    The local coordinates are defined by the posed body mesh (consists of vertcs and faces)

    :param verts: [batch, Nverts, 3]
    :param faces: [uv_size, uv_size, 3], uv_size =e.g. 32
    
    :return: [batch, uv_size, uv_size, 3,3], per example a map of 3x3 rot matrices for local->global transform

    NOTE: local coords are NOT cartesian! uu an vv axis are edges of the triangle,
          not perpendicular (more like barycentric coords)
    '''
    tris = verts[:, faces] # [batch, uv_size, uv_size, 3, 3]
    v1, v2, v3 = tris[:, :, :, 0, :], tris[:, :, :, 1, :], tris[:, :, :, 2, :]
    uu = v2 - v1 # u axis of local coords is the first edge, [batch, uv_size, uv_size, 3]
    vv = v3 - v1 # v axis, second edge
    ww_raw = torch.cross(uu, vv, dim=-1)
    ww = F.normalize(ww_raw, p=2, dim=-1) # unit triangle normal as w axis
    ww_norm = (torch.norm(uu, dim=-1).mean(-1).mean(-1) + torch.norm(vv, dim=-1).mean(-1).mean(-1)) / 2.
    ww = ww*ww_norm.view(len(ww_norm),1,1,1)
    
    # shape of transf_mtx will be [batch, uv_size, uv_size, 3, 3], where the last two dim is like:
    #  |   |   |
    #[ uu  vv  ww]
    #  |   |   |
    # for local to global, say coord in the local coord system is (r,s,t)
    # then the coord in world system should be r*uu + s*vv+ t*ww
    # so the uu, vv, ww should be colum vectors of the local->global transf mtx
    # so when stack, always stack along dim -1 (i.e. column)
    transf_mtx = torch.stack([uu, vv, ww], dim=-1)

    return transf_mtx


def gen_transf_mtx_from_vtransf(vtransf, bary_coords, faces, scaling=1.0):
    '''
    interpolate the local -> global coord transormation given such transformations defined on 
    the body verts (pre-computed) and barycentric coordinates of the query points from the uv map.

    Note: The output of this function, i.e. the transformation matrix of each point, is not a pure rotation matrix (SO3).
    
    args:
        vtransf: [batch, #verts, 3, 3] # per-vertex rotation matrix
        bary_coords: [uv_size, uv_size, 3] # barycentric coordinates of each query point (pixel) on the query uv map 
        faces: [uv_size, uv_size, 3] # the vert id of the 3 vertices of the triangle where each uv pixel locates

    returns: 
        [batch, uv_size, uv_size, 3, 3], transformation matrix for points on the uv surface
    '''
    #  
    vtransf_by_tris = vtransf[:, faces] # shape will be [batch, uvsize, uvsize, 3, 3, 3], where the the last 2 dims being the transf (pure rotation) matrices, the other "3" are 3 points of each triangle
    transf_mtx_uv_pts = torch.einsum('bpqijk,pqi->bpqjk', vtransf_by_tris, bary_coords) # [batch, uvsize, uvsize, 3, 3], last 2 dims are the rotation matix
    transf_mtx_uv_pts *= scaling
    return transf_mtx_uv_pts


class SampleSquarePoints():
    def __init__(self, npoints=16, min_val=0, max_val=1, device='cuda', include_end=True):
        super(SampleSquarePoints, self).__init__()
        self.npoints = npoints
        self.device = device
        self.min_val = min_val # -1 or 0
        self.max_val = max_val # -1 or 0
        self.include_end = include_end

    def sample_regular_points(self, N=None):
        steps = int(self.npoints ** 0.5) if N is None else int(N ** 0.5)
        if self.include_end:
            linspace = torch.linspace(self.min_val, self.max_val, steps=steps) # [0,1]
        else:
            linspace = torch.linspace(self.min_val, self.max_val, steps=steps+1)[: steps] # [0,1)
        grid = torch.stack(torch.meshgrid([linspace, linspace]), -1).to(self.device) #[steps, steps, 2]
        grid = grid.view(-1,2).unsqueeze(0) #[B, N, 2]
        grid.requires_grad = True

        return grid

    def sample_random_points(self, N=None):
        npt = self.npoints if N is None else N
        shape = torch.Size((1, npt, 2))
        rand_grid = torch.Tensor(shape).float().to(self.device)
        rand_grid.data.uniform_(self.min_val, self.max_val)
        rand_grid.requires_grad = True #[B, N, 2]
        return rand_grid


class Embedder():
    '''
    Simple positional encoding, adapted from NeRF: https://github.com/bmild/nerf
    '''
    def __init__(self, **kwargs):

        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):

        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0, input_dims=3):
    '''
    Helper function for positional encoding, adapted from NeRF: https://github.com/bmild/nerf
    '''
    if i == -1:
        return nn.Identity(), input_dims

    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class PositionalEncoding():
    def __init__(self, input_dims=2, num_freqs=10, include_input=False):
        super(PositionalEncoding,self).__init__()
        self.include_input = include_input
        self.num_freqs = num_freqs
        self.input_dims = input_dims

    def create_embedding_fn(self):
        embed_fns = []
        out_dim = 0
        if self.include_input:
            embed_fns.append(lambda x: x)
            out_dim += self.input_dims

        freq_bands = 2. ** torch.linspace(0, self.num_freqs-1, self.num_freqs)
        periodic_fns = [torch.sin, torch.cos]

        for freq in freq_bands:
            for p_fn in periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq:p_fn(math.pi * x * freq))
                # embed_fns.append(lambda x, p_fn=p_fn, freq=freq:p_fn(x * freq))
                out_dim += self.input_dims

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self,coords):
        '''
        use periodic positional encoding to transform cartesian positions to higher dimension
        :param coords: [N, 3]
        :return: [N, 3*2*num_freqs], where 2 comes from that for each frequency there's a sin() and cos()
        '''
        return torch.cat([fn(coords) for fn in self.embed_fns], dim=-1)


def normalize_uv(uv):
    '''
    normalize uv coords from range [0,1] to range [-1,1]
    '''
    return uv * 2. - 1.


def uv_to_grid(uv_idx_map, resolution):
    '''
    uv_idx_map: shape=[batch, N_uvcoords, 2], ranging between 0-1
    this function basically reshapes the uv_idx_map and shift its value range to (-1, 1) (required by F.gridsample)
    the sqaure of resolution = N_uvcoords
    '''
    bs = uv_idx_map.shape[0]
    grid = uv_idx_map.reshape(bs, resolution, resolution, 2) * 2 - 1.
    grid = grid.transpose(1,2)
    return grid