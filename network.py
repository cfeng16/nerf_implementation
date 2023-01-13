import torch
import torch.nn as nn

class nerfmodel(nn.Module):
    def __init__(self, pos_dim, dir_dim, layers=8, width=256, skip=4):
        super().__init__()
        self.net = nn.ModuleList([nn.Linear(pos_dim, width)])
        self.layers = layers
        self.skip = skip
        for i in range(1, self.layers, 1):
            if i == self.skip:
                self.net.append(nn.Linear(pos_dim + width, width))
            elif i != self.layers:
                self.net.append(nn.Linear(width, width))
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.net_last = nn.Linear(width, width)
        self.sigma = nn.Linear(width, 1)
        self.rgb = nn.ModuleList([nn.Linear(dir_dim + width, width // 2)])
        self.rgb.append(nn.Linear(width // 2, 3))
    def forward(self, input_pts, dir):
        pts_copy = input_pts.clone()
        for i in range(self.layers):
            if i == 0:
                feature = self.relu(self.net[i](input_pts))
            elif i == self.skip:
                feature = self.relu(self.net[i](torch.cat((feature, pts_copy), dim=-1)))
            else:
                feature = self.relu(self.net[i](feature))
        #density = self.relu(self.sigma(feature))
        density = self.sigma(feature)
        feature = self.net_last(feature)
        rgb_feature = torch.cat((feature, dir), dim=-1)
        for rgb_sub_layer in self.rgb:
            rgb_feature = rgb_sub_layer(rgb_feature)
        #rgb = self.sigmoid(self.rgb(rgb_feature))
        return density, rgb_feature
    
        

