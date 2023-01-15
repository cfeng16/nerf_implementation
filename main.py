import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from network import nerfmodel
from positional_encoding import positionalencoder
from sampling import stratified_sampling, hierarachical_sampling, rays_sampling
from tqdm import tqdm
import argparse
from graphics import render_from_nerf
import wandb
from torchmetrics import PeakSignalNoiseRatio
from PIL import Image

# wandb.init(project="nerf_implementation")

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
    image =images[i]
    height = image.shape[0]
    width = image.shape[1]
    pose = poses[i]
    rays_o, rays_d = rays_sampling(H=height, W=width, F=focal, c2w=pose)
    rays_o_list.append(torch.flatten(rays_o, start_dim=0, end_dim=1))
    rays_d_list.append(torch.flatten(rays_d, start_dim=0, end_dim=1))
    rays_rgb_list.append(torch.flatten(image, start_dim=0, end_dim=1))

rays_o = torch.cat(rays_o_list, dim=0)
rays_d = torch.cat(rays_d_list, dim=0)
rays_rgb = torch.cat(rays_rgb_list, dim=0)




height, width = testimg.shape[0:2]
test_rays_o, test_rays_d = rays_sampling(H=height, W=width, F=focal, c2w=testpose)
# test_rays_o = torch.flatten(test_rays_o, start_dim=0, end_dim=1)
# test_rays_d = torch.flatten(test_rays_d, start_dim=0, end_dim=1)
test_rgb = testimg.to(device)
#test_rgb = torch.flatten(test_rgb, start_dim=0, end_dim=1)


N = rays_o.shape[0]
bs = 4096
iterations = N // bs




coarse_model = nerfmodel(pos_dim=60, dir_dim=24, layers=8, width=256, skip=4).to(device)
#fine_model = nerfmodel(pos_dim=60, dir_dim=24, layers=8, width=256, skip=4).to(device)
optimizer = torch.optim.RAdam(coarse_model.parameters(), lr=5e-4)

pts_pe = positionalencoder(L=10)
dir_pe = positionalencoder(L=4)
criterion = nn.MSELoss()

epochs = 20
n_samples = 64
n_samples_hierarchical = 64

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
            sigma, rgb = coarse_model(query_points_flat, batches_viewdirs_flat)
            sigma = sigma.view(query_points.shape[:-1])
            rgb = rgb.view(list(query_points.shape[:-1]) + [3])
            rgb_map, _, _, weights = render_from_nerf(nerf_sigma=sigma, nerf_rgb=rgb, z_vals=z_vals, rays_d=rays_d_batch, noise_std=0, device=device)
            # new_query_points, z_vals_combined, new_z_samples = hierarachical_sampling(rays_o=rays_o_batch, rays_d=rays_d_batch, z_vals=z_vals, weights=weights, n_samples=n_samples_hierarchical, device=device)
            # new_query_points_flat = torch.flatten(new_query_points, start_dim=0, end_dim=1)
            # new_query_points_flat = pts_pe(new_query_points_flat)
            # new_batches_viewdirs = rays_d_batch[:, None, ...].expand(new_query_points.shape)
            # new_batches_viewdirs_flat = torch.flatten(new_batches_viewdirs, start_dim=0, end_dim=1)
            # new_batches_viewdirs_flat = F.normalize(new_batches_viewdirs_flat, p=2, dim=-1)
            # new_batches_viewdirs_flat = dir_pe(new_batches_viewdirs_flat)
            # new_sigma, new_rgb = fine_model(new_query_points_flat, new_batches_viewdirs_flat)
            # new_sigma = new_sigma.reshape(new_query_points.shape[:-1])
            # new_rgb = new_rgb.reshape(list(new_query_points.shape[:-1]) + [3])
            # rgb_map_new, _, _, _ = render_from_nerf(nerf_sigma=new_sigma, nerf_rgb=new_rgb, z_vals=z_vals_combined, rays_d=rays_d_batch, noise_std=1, device=device)
            loss = criterion(rgb_map, rays_rgb_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #psnr = -10. * torch.log10(loss)
            pbar.set_postfix({'loss': '{0:1.5f}'.format(loss.item())})
            pbar.update(1)
        with torch.no_grad():
            test_rgb_list = []
            for k in range(test_rays_d.shape[0]):
            #     rays_od = (test_rays_o[k], test_rays_d[k])
            #     rgb, _, __ = render_rays(coarse_model, rays_od, bound=(2.,6.), N_samples=(n_samples,None), device=device, use_view=True)
            #     rgb_list.append(rgb.unsqueeze(0))
            # rgb = torch.cat(rgb_list, dim=0)
            # loss = criterion(rgb, torch.tensor(test_rgb, device=device)).cpu()
            # psnr = -10. * torch.log(loss).item() / torch.log(torch.tensor([10.]))
            # print(f"PSNR={psnr.item()}")
                rays_o_batch = test_rays_o[k]
                rays_d_batch = test_rays_d[k]
                query_points, z_vals = stratified_sampling(rays_o=rays_o_batch, rays_d=rays_d_batch, near=near, far=far, n_samples=n_samples, device=device)
                batches_viewdirs = rays_d_batch[:, None, ...].expand(query_points.shape)
                query_points_flat = torch.flatten(query_points, start_dim=0, end_dim=1)
                batches_viewdirs_flat = torch.flatten(batches_viewdirs, start_dim=0, end_dim=1)
                batches_viewdirs_flat = F.normalize(batches_viewdirs_flat, p=2, dim=-1)
                query_points_flat = pts_pe(query_points_flat)
                batches_viewdirs_flat = dir_pe(batches_viewdirs_flat)
                sigma, rgb = coarse_model(query_points_flat, batches_viewdirs_flat)
                sigma = sigma.view(query_points.shape[:-1])
                rgb = rgb.view(list(query_points.shape[:-1]) + [3])
                rgb_map, _, _, weights = render_from_nerf(nerf_sigma=sigma, nerf_rgb=rgb, z_vals=z_vals, rays_d=rays_d_batch, noise_std=0, device=device)
                # new_query_points, z_vals_combined, new_z_samples = hierarachical_sampling(rays_o=rays_o_batch, rays_d=rays_d_batch, z_vals=z_vals, weights=weights, n_samples=n_samples_hierarchical, device=device)
                # new_query_points_flat = torch.flatten(new_query_points, start_dim=0, end_dim=1)
                # new_query_points_flat = pts_pe(new_query_points_flat)
                # new_batches_viewdirs = rays_d_batch[:, None, ...].expand(new_query_points.shape)
                # new_batches_viewdirs_flat = torch.flatten(new_batches_viewdirs, start_dim=0, end_dim=1)
                # new_batches_viewdirs_flat = F.normalize(new_batches_viewdirs_flat, p=2, dim=-1)
                # new_batches_viewdirs_flat = dir_pe(new_batches_viewdirs_flat)
                # new_sigma, new_rgb = fine_model(new_query_points_flat, new_batches_viewdirs_flat)
                # new_sigma = new_sigma.reshape(new_query_points.shape[:-1])
                # new_rgb = new_rgb.reshape(list(new_query_points.shape[:-1]) + [3])
                # rgb_map_new, _, _, _ = render_from_nerf(nerf_sigma=new_sigma, nerf_rgb=new_rgb, z_vals=z_vals_combined, rays_d=rays_d_batch, noise_std=1, device=device)
                test_rgb_list.append(rgb_map.unsqueeze(0))
            test_rgb_pred = torch.cat(test_rgb_list, dim=0)
            test_rgb_pred = test_rgb_pred
            loss = criterion(test_rgb_pred, test_rgb).cpu()
            psnr = -10. * torch.log10(loss)
            print(f"PSNR={psnr.item()}")
            test_img_vis = Image.fromarray((test_rgb_pred*255).cpu().numpy().astype('uint8'))
            test_img_vis.save('/home/chfeng/nerf_implementation/pred.png')
            #pbar.set_postfix({'PSNR': '{0:2.2f}'.format(psnr.item())})
