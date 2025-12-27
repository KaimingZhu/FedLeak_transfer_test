import torch

@torch.no_grad()
def normalization(imgs, scale_factor=10.):
    imgs_norm = imgs.view(imgs.size(0), -1).norm(2, dim=1).view(-1, 1, 1, 1) + 1e-6
    normalized_imgs = imgs / imgs_norm
    scale_factor = 1. / (normalized_imgs.view(imgs.size(0), -1).max(dim=1)[0] + 1e-6)
    return normalized_imgs * scale_factor.view(-1, 1, 1, 1)

@torch.no_grad()
def compute_psnr(imgs1, imgs2, is_normalization):
    if is_normalization:
        x, y = normalization(imgs1), normalization(imgs2)
    else:
        x, y = imgs1, imgs2
    mse = ((x - y) ** 2).mean(dim=[1, 2, 3])
    return 10 * torch.log10(1 / (mse+1e-6))


@torch.no_grad()
def update_best_ones(best_ones, original_ones, recovered_ones, is_normalization=False):
    metric_fn = compute_psnr
    res = [metric_fn(original_ones[i].unsqueeze(0), recovered_ones, is_normalization) for i in range(original_ones.size(0))]
    res = torch.stack(res, dim=0)
    vals, idxs = res.max(dim=1)
    mask = metric_fn(best_ones, original_ones, is_normalization) < vals
    mask = mask.view(-1, 1, 1, 1)
    mask = mask.to(recovered_ones[idxs].device)
    return recovered_ones[idxs] * mask.float() + best_ones * (~mask).float()

