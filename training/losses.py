import torch

def smooth_labels(y, smooth=0.05):
    return y * (1 - smooth) + smooth * 0.5

def r1_regularization(d_out_real, real_imgs):
    grads = torch.autograd.grad(
        outputs=d_out_real.sum(),
        inputs=real_imgs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    return grads.view(grads.size(0), -1).pow(2).sum(dim=1).mean()
