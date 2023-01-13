import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def render_from_nerf(nerf_sigma, nerf_rgb, z_vals, rays_d, noise_std, device):
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, 1e10 * torch.ones_like(dists[..., :1])], dim=-1)
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    noise = torch.randn(nerf_sigma.shape, device=device)*noise_std
    sigma = F.relu(noise + nerf_sigma)
    alpha = 1 - torch.exp(-sigma * dists)
    transmittence = cumprod_exclusive(1. - alpha + 1e-10)
    weights = transmittence * alpha 
    rgb_map = torch.sum(weights[..., None] * nerf_rgb, dim=-2)
    depth_map = torch.sum(weights * z_vals, dim=-1)
    acc_map = torch.sum(weights, dim=-1)
    return rgb_map, depth_map, acc_map, weights

def cumprod_exclusive(
  tensor: torch.Tensor
) -> torch.Tensor:
  r"""
  (Courtesy of https://github.com/krrish94/nerf-pytorch)

  Mimick functionality of tf.math.cumprod(..., exclusive=True), as it isn't available in PyTorch.

  Args:
  tensor (torch.Tensor): Tensor whose cumprod (cumulative product, see `torch.cumprod`) along dim=-1
    is to be computed.
  Returns:
  cumprod (torch.Tensor): cumprod of Tensor along dim=-1, mimiciking the functionality of
    tf.math.cumprod(..., exclusive=True) (see `tf.math.cumprod` for details).
  """

  # Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
  cumprod = torch.cumprod(tensor, -1)
  # "Roll" the elements along dimension 'dim' by 1 element.
  cumprod = torch.roll(cumprod, 1, -1)
  # Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
  cumprod[..., 0] = 1.
  
  return cumprod



if __name__ == "__main__":
    rays_o = torch.rand((25,1))
    rays_d = torch.rand((25,3))
    z_vals = torch.rand((25, 100))
    near = 2
    far = 6
    n_samples=20
    render_from_nerf(rays_o, rays_d,z_vals,rays_d, noise_std=1 )
    print()