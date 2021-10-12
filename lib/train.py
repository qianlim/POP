import torch

from lib.losses import normal_loss, chamfer_loss_separate
from lib.utils_model import gen_transf_mtx_from_vtransf


def train(
        model, geom_featmap, train_loader, optimizer,
        flist_uv=None, valid_idx=None, uv_coord_map=None,
        bary_coords_map=None,
        device='cuda',
        subpixel_sampler=None,
        loss_weights=None,
        transf_scaling=1.0,
        ):

    n_train_samples = len(train_loader.dataset)

    train_s2m, train_m2s, train_lnormal, train_rgl, train_latent_rgl, train_total = 0, 0, 0, 0, 0, 0
    w_s2m, w_m2s, w_normal, w_rgl, w_latent_rgl = loss_weights

    model.train()
    for _, data in enumerate(train_loader):

        # -------------------------------------------------------
        # ------------ load batch data and reshaping ------------

        # query_posmap and inp_posmap are in the same pose; 
        # query_posmap: in each subject's body shape, used to add to the predicted clothing displacements
        # inp_posmap: in the SMPL/SMPLX default template body shape, used to extract shape-agnostic pose features
        [query_posmap, inp_posmap, target_pc_n, target_pc, vtransf, target_names, body_verts, index] = data 

        gpu_data = [query_posmap, inp_posmap, target_pc_n, target_pc, vtransf, body_verts, index]
        [query_posmap, inp_posmap, target_pc_n, target_pc, vtransf, body_verts, index] = list(map(lambda x: x.to(device), gpu_data))
        bs, _, H, W = query_posmap.size()

        optimizer.zero_grad()

        transf_mtx_map = gen_transf_mtx_from_vtransf(vtransf, bary_coords_map, flist_uv, scaling=transf_scaling)

        # original geom feat map: [num_outfits, C, H, W]
        # each clotype (the 'index' when loading the data) uses a unique [C, H, W] slice for all its frames
        geom_featmap_batch = geom_featmap[index, ...]

        uv_coord_map_batch = uv_coord_map.expand(bs, -1, -1).contiguous()

        # for compatibility with the SCALE code (https://github.com/qianlim/SCALE) we keep the 'patch sampling' notion, 
        # but only sample 1 point per 'patch'; i.e. there's no 'patch' any more, just points, and the 'patch coordinates' (p,q) is always (0,0)
        pq_samples = subpixel_sampler.sample_regular_points()
        pq_repeated = pq_samples.expand(bs, H * W, -1, -1) # repeat the same pq parameterization for all patches

        N_subsample = 1
        bp_locations = query_posmap.expand(N_subsample, -1, -1,-1,-1).permute([1, 2, 3, 4, 0]) #[bs, C, H, W, N_sample],
        transf_mtx_map = transf_mtx_map.expand(N_subsample, -1, -1, -1, -1, -1).permute([1, 2, 3, 0, 4, 5])  # [bs, H, W, N_subsample, 3, 3]

        # --------------------------------------------------------------------
        # ------------ model pass an coordinate transformation ---------------

        # Core: predict the clothing residual (displacement) from the body, and their normals
        pred_res, pred_normals = model(inp_posmap, geom_featmap=geom_featmap_batch,
                                       uv_loc=uv_coord_map_batch,
                                       pq_coords=pq_repeated)

        # local coords --> global coords
        pred_res = pred_res.permute([0,2,3,4,1]).unsqueeze(-1)
        pred_normals = pred_normals.permute([0,2,3,4,1]).unsqueeze(-1)

        pred_res = torch.matmul(transf_mtx_map, pred_res).squeeze(-1)
        pred_normals = torch.matmul(transf_mtx_map, pred_normals).squeeze(-1)
        pred_normals = torch.nn.functional.normalize(pred_normals, dim=-1)
        
        # residual to abosolute locations in space
        full_pred = pred_res.permute([0,4,1,2,3]).contiguous() + bp_locations

        # take the selected points and reshape to [Npoints, 3]
        full_pred = full_pred.permute([0,2,3,4,1]).reshape(bs, -1, N_subsample, 3)[:, valid_idx, ...]
        pred_normals = pred_normals.reshape(bs, -1, N_subsample, 3)[:, valid_idx, ...]

        # reshaping the points that are grouped into patches into a big point set
        full_pred = full_pred.reshape(bs, -1, 3).contiguous()
        pred_normals = pred_normals.reshape(bs, -1, 3).contiguous()

        # --------------------------------
        # ------------ losses ------------

        # Chamfer dist from the (s)can to (m)odel: from the GT points to its closest ponit in the predicted point set
        m2s, s2m, idx_closest_gt, _ = chamfer_loss_separate(full_pred, target_pc) #idx1: [#pred points]
        s2m = torch.mean(s2m)

        # normal loss
        lnormal, closest_target_normals = normal_loss(pred_normals, target_pc_n, idx_closest_gt)
        
        # dist from the predicted points to their respective closest point on the GT, projected by
        # the normal of these GT points, to appxoimate the point-to-surface distance
        nearest_idx = idx_closest_gt.expand(3, -1, -1).permute([1,2,0]).long() # [batch, N] --> [batch, N, 3], repeat for the last dim
        target_points_chosen = torch.gather(target_pc, dim=1, index=nearest_idx)
        pc_diff = target_points_chosen - full_pred # vectors from prediction to its closest point in gt pcl
        m2s = torch.sum(pc_diff * closest_target_normals, dim=-1) # project on direction of the normal of these gt points
        m2s = torch.mean(m2s**2) # the length (squared) is the approx. pred point to scan surface dist.

        rgl_len = torch.mean(pred_res ** 2)
        rgl_latent = torch.mean(geom_featmap_batch**2)

        loss = s2m*w_s2m + m2s*w_m2s + lnormal* w_normal + rgl_len*w_rgl + rgl_latent*w_latent_rgl

        loss.backward()
        optimizer.step()

        # ------------------------------------------
        # ------------ accumulate stats ------------

        train_m2s += m2s * bs
        train_s2m += s2m * bs
        train_lnormal += lnormal * bs
        train_rgl += rgl_len * bs
        train_latent_rgl += rgl_latent * bs

        train_total += loss * bs

    train_s2m /= n_train_samples
    train_m2s /= n_train_samples
    train_lnormal /= n_train_samples
    train_rgl /= n_train_samples
    train_latent_rgl /= n_train_samples
    train_total /= n_train_samples

    return train_m2s, train_s2m, train_lnormal, train_rgl, train_latent_rgl, train_total

