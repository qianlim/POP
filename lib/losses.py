import torch
import torch.nn.functional as F

def chamfer_loss_separate(output, target, weight=1e4, phase='train', debug=False):
    from chamferdist.chamferdist import ChamferDistance
    cdist = ChamferDistance()
    model2scan, scan2model, idx1, idx2 = cdist(output, target)
    if phase == 'train':
        return model2scan, scan2model, idx1, idx2
    else: # in test, show both directions, average over points, but keep batch
        return torch.mean(model2scan, dim=-1)* weight, torch.mean(scan2model, dim=-1)* weight,


def normal_loss(output_normals, target_normals, nearest_idx, weight=1.0, phase='train'):
    '''
    Given the set of nearest neighbors found by chamfer distance, calculate the
    L1 discrepancy between the predicted and GT normals on each nearest neighbor point pairs.
    Note: the input normals are already normalized (length==1).
    '''
    nearest_idx = nearest_idx.expand(3, -1, -1).permute([1,2,0]).long() # [batch, N] --> [batch, N, 3], repeat for the last dim
    target_normals_chosen = torch.gather(target_normals, dim=1, index=nearest_idx)

    assert output_normals.shape == target_normals_chosen.shape

    if phase == 'train':
        lnormal = F.l1_loss(output_normals, target_normals_chosen, reduction='mean')  # [batch, 8000, 3])
        return lnormal, target_normals_chosen
    else:
        lnormal = F.l1_loss(output_normals, target_normals_chosen, reduction='none')
        lnormal = lnormal.mean(-1).mean(-1) # avg over all but batch axis
        return lnormal, target_normals_chosen


def color_loss(output_colors, target_colors, nearest_idx, weight=1.0, phase='train', excl_holes=False):
    '''
    Similar to normal loss, used in training a color prediction model.
    '''
    nearest_idx = nearest_idx.expand(3, -1, -1).permute([1,2,0]).long() # [batch, N] --> [batch, N, 3], repeat for the last dim
    target_colors_chosen = torch.gather(target_colors, dim=1, index=nearest_idx)

    assert output_colors.shape == target_colors_chosen.shape
    
    if excl_holes:
        # scan holes have rgb all=0, exclude these from supervision
        colorsum = target_colors_chosen.sum(-1)
        mask = (colorsum!=0).float().unsqueeze(-1)
    else:
        mask = 1.

    if phase == 'train':
        lcolor = F.l1_loss(output_colors, target_colors_chosen, reduction='none')  # [batch, 8000, 3])
        lcolor = lcolor * mask
        lcolor = lcolor.mean()
        return lcolor, target_colors_chosen
    else:
        lcolor = F.l1_loss(output_colors, target_colors_chosen, reduction='none')
        lcolor = lcolor * mask
        lcolor = lcolor.mean(-1).mean(-1) # avg over all but batch axis
        return lcolor, target_colors_chosen