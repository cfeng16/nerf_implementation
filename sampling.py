import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def sample_rays(H, W, F, c2w):
    i, j = torch.meshgrid(torch.arange(W, dtype=torch.float32).to(c2w), \
        torch.arange(H, dtype=torch.float32).to(c2w), indexing='xy')
    dirs = torch.stack([(i - W * .5) / F, -(j - H * .5) / F, -torch.ones_like(i)], dim=-1)
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d

def stratified_sampling(rays_o, rays_d, near, far, n_samples, device):
    t_vals = torch.linspace(0.0, 1.0, n_samples, device=device)
    z_vals = near * (1 - t_vals) + far * t_vals
    mids = (z_vals[1:] + z_vals[:-1]) / 2
    upper = torch.cat((mids, z_vals[-1:]), dim=0)
    lower = torch.cat((z_vals[:1], mids), dim=0)
    t_rand = torch.rand([n_samples], device=device)
    z_vals = lower + (upper-lower)*t_rand
    z_vals = z_vals.expand(list(rays_o.shape[:-1]) + [n_samples])
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    return pts, z_vals

def sample_pdf(bins, weights, n_sample, device):
    pdf = (weights + 1e-5) / torch.sum(weights + 1e-5, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat((torch.zeros_like(cdf[..., :1]), cdf), dim=-1)
    u = torch.rand(list(cdf.shape[:-1]) + [n_sample], device=device).contiguous()
    ids = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp(ids - 1, min=0)
    above = torch.clamp(ids, max = cdf.shape[-1] - 1)
    ids_g = torch.stack([below, above], dim=-1)
    matched_shape = [ids_g.shape[0], ids_g.shape[1], cdf.shape[-1]]
    cdf_val = torch.gather(cdf.unsqueeze(1).expand(matched_shape), dim=-1, index=ids_g)
    bins_val = torch.gather(bins[None, None, :].expand(matched_shape), dim=-1, index=ids_g)
    cdf_d = (cdf_val[..., 1] - cdf_val[..., 0])   
    cdf_d = torch.where(cdf_d < 1e-5, torch.ones_like(cdf_d, device=device), cdf_d)
    t = (u - cdf_val[..., 0]) / cdf_d
    samples = bins_val[..., 0] + t * (bins_val[..., 1] - bins_val[..., 0])
    return samples


def hierarachical_sampling(rays_o, rays_d, z_vals, weights, n_samples):
    z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    new_z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], n_samples)
    new_z_samples = new_z_samples.detach()
    z_vals_combined, _ = torch.sort(torch.cat([z_vals, new_z_samples], dim=-1), dim=-1)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_combined[..., :, None]  # [N_rays, N_samples + n_samples, 3]
    return pts, z_vals_combined, new_z_samples



# if __name__ == "__main__":
#     rays_o = torch.rand((25,3))
#     rays_d = torch.rand((25,3))
#     near = 2
#     far = 6
#     n_samples=20
#     stratified_sampling(rays_o, rays_d, near, far, n_samples, device="cpu")
#     print()



