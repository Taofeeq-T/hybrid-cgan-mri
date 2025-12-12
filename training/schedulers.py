import re

def set_requires_grad(module, requires_grad: bool, name_filter=None):
    for n, p in module.named_parameters():
        if name_filter is None or re.search(name_filter, n):
            p.requires_grad = requires_grad

def unfreeze_vit_last_blocks(discriminator, num_blocks=1):
    vit = discriminator.vit
    set_requires_grad(vit, False)

    total = len(vit.blocks)
    for i in range(total - num_blocks, total):
        set_requires_grad(vit.blocks[i], True)

    if hasattr(vit, "norm"):
        set_requires_grad(vit.norm, True)
