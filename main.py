import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from network import nerfmodel
from positional_encoding import positionalencoder
from sampling import stratified_sampling, hierarachical_sampling, rays_sampling
from tqdm import tqdm
import argparse

device = torch.device("cuda:0")
data = np.load('tiny_nerf_data.npz')
images = data['images']
poses = data['poses']
focal = data['focal']
images = torch.from_numpy(images).to(device)
poses = torch.from_numpy(poses).to(device)
focal = torch.from_numpy(focal).to(device)

height, width = images.shape[1:3]
near, far = 2.0, 6.0

n_training = 100
testimg_idx = 101

testimg, testpose = images[testimg_idx], poses[testimg_idx]

train_images = images[:n_training]
poses = poses[:n_training]

rays_o_list = []
rays_d_list = []
rays_rgb_list = []

for i in range(n_training):
    image = images[i]
    pose = poses[i]
    rays_o, rays_d = rays_sampling(H=height, W=width, F=focal, c2w=pose)
    rays_o_list.append(torch.flatten(rays_o, start_dim=0, end_dim=1))
    rays_d_list.append(torch.flatten(rays_d, start_dim=0, end_dim=1))
    rays_rgb_list.append(torch.flatten(image, start_dim=0, end_dim=1))

rays_o = torch.cat(rays_o_list, dim=0)
rays_d = torch.cat(rays_d_list, dim=0)
rays_rgb = torch.cat(rays_rgb_list, dim=0)

N = rays_o.shape[0]
bs = 4096
iterations = N // bs

test_rays_o, test_rays_d = rays_sampling(H=height, W=width, F=focal, c2w=testpose)


coarse_model = nerfmodel(pos_dim=63, dir_dim=27, layers=8, width=256, skip=4).to(device)
fine_model = nerfmodel(pos_dim=63, dir_dim=27, layers=8, width=256, skip=4).to(device)
coarse_optimizer = torch.optim.Adam(coarse_model.parameters(), lr=5e-4)
fine_optimizer = torch.optim.Adam(fine_model.parameters(), lr=5e-4)
pts_pe = positionalencoder(L=10)
dir_pe = positionalencoder(L=4)
criterion = nn.MSELoss()

epochs = 10
n_samples = 64

for epoch in range(epochs):
    train_index = torch.randperm(N)
    rays_o_train = rays_o[train_index, :]
    rays_d_train = rays_d[train_index, :]
    rays_rgb_train = rays_rgb[train_index, :]
    rays_o_iter = iter(torch.split(rays_o_train, bs, dim=0))
    rays_d_iter = iter(torch.split(rays_d_train, bs, dim=0))
    rays_rgb_iter = iter(torch.split(rays_rgb_train, bs, dim=0))
    with tqdm(total=iterations, desc="Epoch {}".format(epoch), ncols=80) as pbar:
        for i in range(iterations):
            rays_o_batch = next(rays_o_iter)
            rays_d_batch = next(rays_d_iter)
            rays_rgb_batch = next(rays_rgb_iter)
            query_points, z_vals = stratified_sampling(rays_o=rays_o_batch, rays_d=rays_d_batch, near=near, far=far, n_samples=n_samples, device=device)
            batches_viewdirs = rays_d_batch[:, None, ...].expand(query_points.shape)
            query_points_flat = torch.flatten(query_points, start_dim=0, end_dim=1)
            batches_viewdirs_flat = torch.flatten(batches_viewdirs, start_dim=0, end_dim=1)
            batches_viewdirs_flat = F.normalize(batches_viewdirs_flat, p=2, dim=-1)
            query_points_flat = pts_pe(query_points_flat)
            batches_viewdirs_flat = dir_pe(batches_viewdirs_flat)
