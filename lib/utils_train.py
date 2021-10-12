def adjust_loss_weights(init_weight, current_epoch, mode='decay', start=400, every=20):
    # decay or rise the loss weights according to the given policy and current epoch
    # mode: decay, rise or binary

    if mode != 'binary':
        if current_epoch < start:
            if mode == 'rise':
                weight = init_weight * 1e-6 # use a very small weight for the normal loss in the beginning until the chamfer dist stabalizes
            else:
                weight = init_weight
        else:
            if every == 0:
                weight = init_weight # don't rise, keep const
            else:
                if mode == 'rise':
                    weight = init_weight * (1.05 ** ((current_epoch - start) // every))
                else:
                    weight = init_weight * (0.85 ** ((current_epoch - start) // every))

    return weight