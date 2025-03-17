# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os,sys
import numpy as np
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)
sys.path.append(f'{code_dir}/../../../../')
from Utils import *
import torch.nn.functional as F
import torch
import torch.nn as nn
import cv2
from functools import partial
from network_modules import *
from Utils import *
import kornia
canny = kornia.filters.Canny()



class RefineNet(nn.Module):
  def __init__(self, cfg=None, c_in=4, n_view=1):
    super().__init__()
    self.cfg = cfg
    if self.cfg.use_BN:
      norm_layer = nn.BatchNorm2d
      norm_layer1d = nn.BatchNorm1d
    else:
      norm_layer = None
      norm_layer1d = None

    self.encodeA = nn.Sequential(
      ConvBNReLU(C_in=c_in,C_out=64,kernel_size=7,stride=2, norm_layer=norm_layer),
      ConvBNReLU(C_in=64,C_out=128,kernel_size=3,stride=2, norm_layer=norm_layer),
      ResnetBasicBlock(128,128,bias=True, norm_layer=norm_layer),
      ResnetBasicBlock(128,128,bias=True, norm_layer=norm_layer),
    )

    self.encodeAB = nn.Sequential(
      ResnetBasicBlock(256,256,bias=True, norm_layer=norm_layer),
      ResnetBasicBlock(256,256,bias=True, norm_layer=norm_layer),
      ConvBNReLU(256,512,kernel_size=3,stride=2, norm_layer=norm_layer),
      ResnetBasicBlock(512,512,bias=True, norm_layer=norm_layer),
      ResnetBasicBlock(512,512,bias=True, norm_layer=norm_layer),
    )

    embed_dim = 512
    num_heads = 4
    self.pos_embed = PositionalEmbedding(d_model=embed_dim, max_len=400)

    self.trans_head = nn.Sequential(
      nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=512, batch_first=True),
		  nn.Linear(512, 3),
    )

    if self.cfg['rot_rep']=='axis_angle':
      rot_out_dim = 3
    elif self.cfg['rot_rep']=='6d':
      rot_out_dim = 6
    else:
      raise RuntimeError
    self.rot_head = nn.Sequential(
      nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=512, batch_first=True),
		  nn.Linear(512, rot_out_dim),
    )


  def forward(self, A, B):
    """
    @A: (B,C,H,W)
    """
    bs = len(A)
    output = {}

    x = torch.cat([A,B], dim=0)
    x = self.encodeA(x)
    a = x[:bs]
    b = x[bs:]

    ab = torch.cat((a,b),1).contiguous()
    ab = self.encodeAB(ab)  #(B,C,H,W)

    ab = self.pos_embed(ab.reshape(bs, ab.shape[1], -1).permute(0,2,1))

    output['trans'] = self.trans_head(ab).mean(dim=1)
    output['rot'] = self.rot_head(ab).mean(dim=1)

    # return output
    return output,x


#######################################################  zero shot rgb refine ################################################################
class RefineNet_RGB(nn.Module):
  def __init__(self, cfg=None, c_in=3, n_view=1):
    super().__init__()
    self.cfg = cfg
    if self.cfg.use_BN:
      norm_layer = nn.BatchNorm2d
      norm_layer1d = nn.BatchNorm1d
    else:
      norm_layer = None
      norm_layer1d = None

    self.encodeA = nn.Sequential(
      ConvBNReLU(C_in=c_in,C_out=64,kernel_size=7,stride=2, norm_layer=norm_layer),
      ConvBNReLU(C_in=64,C_out=128,kernel_size=3,stride=2, norm_layer=norm_layer),
      ResnetBasicBlock(128,128,bias=True, norm_layer=norm_layer),
      ResnetBasicBlock(128,128,bias=True, norm_layer=norm_layer),
    )

    self.encodeAB = nn.Sequential(
      ResnetBasicBlock(256,256,bias=True, norm_layer=norm_layer),
      ResnetBasicBlock(256,256,bias=True, norm_layer=norm_layer),
      ConvBNReLU(256,512,kernel_size=3,stride=2, norm_layer=norm_layer),
      ResnetBasicBlock(512,512,bias=True, norm_layer=norm_layer),
      ResnetBasicBlock(512,512,bias=True, norm_layer=norm_layer),
    )

    embed_dim = 512
    num_heads = 4
    self.pos_embed = PositionalEmbedding(d_model=embed_dim, max_len=400)

    self.trans_head = nn.Sequential(
      nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=512, batch_first=True),
		  nn.Linear(512, 3),
    )

    if self.cfg['rot_rep']=='axis_angle':
      rot_out_dim = 3
    elif self.cfg['rot_rep']=='6d':
      rot_out_dim = 6
    else:
      raise RuntimeError
    self.rot_head = nn.Sequential(
      nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=512, batch_first=True),
		  nn.Linear(512, rot_out_dim),
    )


  def forward(self, A, B):
    """
    @A: (B,C,H,W)
    """
    A = A[:,:3,:,:]
    B = B[:,:3,:,:]

    _, A_canny = canny(A)
    _, B_canny = canny(B)

    # show_a = A_canny[0].data.cpu().numpy()
    # show_b = B_canny[0].data.cpu().numpy()
    # cv2.imshow('show_a',show_a[0])
    # cv2.imshow('show_b',show_b[0])
    # cv2.waitKey()


    A = torch.cat([A,A_canny], dim=1)
    B = torch.cat([B,B_canny], dim=1)


    bs = len(A)
    output = {}

    x = torch.cat([A,B], dim=0)
    x = self.encodeA(x)
    a = x[:bs]
    b = x[bs:]

    ab = torch.cat((a,b),1).contiguous()
    ab = self.encodeAB(ab)  #(B,C,H,W)

    ab = self.pos_embed(ab.reshape(bs, ab.shape[1], -1).permute(0,2,1))

    output['trans'] = self.trans_head(ab).mean(dim=1)
    output['rot'] = self.rot_head(ab).mean(dim=1)

    return output,x








#
# from omegaconf import OmegaConf
# run_name = "2023-10-28-18-33-37"
# cfg = OmegaConf.load(f'/home/sunh/6D_ws/Fpose_rgb/weights/{run_name}/config.yml')
# model = RefineNet(cfg=cfg, c_in=cfg['c_in']).cuda()
# model_name = 'model_best.pth'
# ckpt_dir = f'/home/sunh/6D_ws/Fpose_rgb/weights/{run_name}/{model_name}'
# ckpt = torch.load(ckpt_dir)
# if 'model' in ckpt:
#   ckpt = ckpt['model']
# model.load_state_dict(ckpt)
# model.cuda().eval()
#
# def process_rgbd(rgb,depth):
#   rgb_tensor = torch.as_tensor(rgb[:,:480,:], device='cuda', dtype=torch.float)
#   depth = torch.as_tensor(depth[:,:480], device='cuda', dtype=torch.float)
#   depth = erode_depth(depth, radius=2, device='cuda')
#   depth = bilateral_filter_depth(depth, radius=2, device='cuda')
#   logging.info("depth processing done")
#   # K = np.array([[615.37, 0, 323.84],
#   #               [0, 615.31, 232.17],
#   #               [0, 0, 1]])
#   K = np.array([[597.11, 0, 325.671],
#                 [0, 597.46, 236.537],
#                 [0, 0, 1]])
#   xyz_map = depth2xyzmap_batch(depth[None], torch.as_tensor(K, dtype=torch.float, device='cuda')[None], zfar=np.inf)[0]
#   xyz_map = xyz_map[:,:480,:]
#   xyz_map_tensor = torch.as_tensor(xyz_map, device='cuda', dtype=torch.float)
#
#   rgb_tensor = rgb_tensor.permute(2, 0, 1).unsqueeze(0)
#   xyz_map_tensor = xyz_map_tensor.permute(2, 0, 1).unsqueeze(0)
#
#   rgb_tensor = F.interpolate(rgb_tensor, size=(160, 160), mode='bilinear', align_corners=False)
#   xyz_map_tensor = F.interpolate(xyz_map_tensor, size=(160, 160), mode='bilinear', align_corners=False)
#
#   A = torch.cat([rgb_tensor.cuda(), xyz_map_tensor.cuda()], dim=1).float()
#   return A
#
#
# ### 6 include rgb xyz(depth2xyz)  in estimater.py
# rgb = cv2.imread('/home/sunh/6D_ws/ActivePose/FoundationPose-main/demo_data/my/rgb/{:06d}-color.png'.format(1))
# depth = cv2.imread('/home/sunh/6D_ws/ActivePose/FoundationPose-main/demo_data/my/depth/{:06d}-depth.png'.format(1),
#                    -1) / 1000.
#
# # rgb = cv2.imread('/home/robotlab/sunhan/FoundationPose-main/data/{}.png'.format(46))
# # depth = np.load('/home/robotlab/sunhan/FoundationPose-main/data/{}.npy'.format(46)) * 0.00025
# # print(depth.shape)
#
# A = process_rgbd(rgb,depth)
# rgb = cv2.imread('/home/sunh/6D_ws/ActivePose/FoundationPose-main/demo_data/my/rgb/{:06d}-color.png'.format(5))
# depth = cv2.imread('/home/sunh/6D_ws/ActivePose/FoundationPose-main/demo_data/my/depth/{:06d}-depth.png'.format(5),
#                    -1) / 1000.
# # rgb = cv2.imread('/home/robotlab/sunhan/FoundationPose-main/data/{}.png'.format(1))
# # depth = np.load('/home/robotlab/sunhan/FoundationPose-main/data/{}.npy'.format(1)) * 0.00025
# B = process_rgbd(rgb,depth)
# print(B.shape)
#
# out = model(A,B)
# # trans = out['trans'].cpu().detach().numpy()
# # rot = out['rot'].cpu().detach().numpy()
# print(out)
# # print(rot)
