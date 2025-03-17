# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import functools
import os,sys,kornia
import time
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../../')
import numpy as np
import torch
from omegaconf import OmegaConf
from learning.models.refine_network import RefineNet, RefineNet_RGB
from learning.datasets.h5_dataset import *
from Utils import *
from datareader import *
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import geomloss

# 提取旋转矩阵和平移向量
def extract_rotation_translation(pose):
    # 提取旋转矩阵（3x3部分）和平移向量（最后一列前三个元素）
    rotation = pose[:, :3, :3]  # 12x3x3
    translation = pose[:, :3, 3]  # 12x3
    return rotation, translation

# 计算旋转和平移残差
def compute_residual(pose, gt_pose, cfg, mesh_diameter):
    # 提取pose1和gt_pose中的旋转矩阵和平移向量
    R1, t1 = extract_rotation_translation(pose)
    R_gt, t_gt = extract_rotation_translation(gt_pose)

    # 计算旋转残差：R_residual = R_gt * R1^-1
    R_residual = torch.matmul(R_gt, torch.linalg.inv(R1) )

    # 将旋转矩阵转换为轴角形式
    axis_angle_residual = rotation_matrix_to_axis_angle(R_residual.permute(0, 2, 1))
    axis_angle_residual = axis_angle_residual /  cfg['rot_normalizer']

    # 计算平移残差：t_residual = t_gt - t1
    translation_residual = t_gt - t1

    # 如果需要归一化平移残差
    if cfg['normalize_xyz']:
        translation_residual = translation_residual/(mesh_diameter / 2)

    return axis_angle_residual, translation_residual

# 示例：将旋转矩阵转换为轴角形式
def rotation_matrix_to_axis_angle(rotation_matrix):
    # 使用torch的so3_log_map或其他库实现
    return so3_log_map(rotation_matrix)



p = 2
entreg = .1 # entropy regularization factor for Sinkhorn
OTLoss = geomloss.SamplesLoss(
    loss='sinkhorn', p=p,
    # 对于p=1或p=2的情形
    cost=geomloss.utils.distances if p==1 else geomloss.utils.squared_distances,
    blur=entreg**(1/p), backend='tensorized')


logging.info("welcome")
amp = True
run_name = "2023-10-28-18-33-37"
model_name = 'model_best.pth'
code_dir = os.path.dirname(os.path.realpath(__file__))
ckpt_dir = f'{code_dir}/weights/{run_name}/{model_name}'

cfg = OmegaConf.load(f'{code_dir}/weights/{ run_name}/config.yml')

cfg['ckpt_dir'] = ckpt_dir
cfg['enable_amp'] = True

########## Defaults, to be backward compatible
if 'use_normal' not in  cfg:
   cfg['use_normal'] = False
if 'use_mask' not in  cfg:
   cfg['use_mask'] = False
if 'use_BN' not in  cfg:
   cfg['use_BN'] = False
if 'c_in' not in  cfg:
   cfg['c_in'] = 4
if 'crop_ratio' not in  cfg or  cfg['crop_ratio'] is None:
   cfg['crop_ratio'] = 1.2
if 'n_view' not in  cfg:
   cfg['n_view'] = 1
if 'trans_rep' not in  cfg:
   cfg['trans_rep'] = 'tracknet'
if 'rot_rep' not in  cfg:
   cfg['rot_rep'] = 'axis_angle'
if 'zfar' not in  cfg:
   cfg['zfar'] = 3
if 'normalize_xyz' not in  cfg:
   cfg['normalize_xyz'] = False
if isinstance( cfg['zfar'], str) and 'inf' in  cfg['zfar'].lower():
   cfg['zfar'] = np.inf
if 'normal_uint8' not in  cfg:
   cfg['normal_uint8'] = False
logging.info(f" cfg: \n {OmegaConf.to_yaml( cfg)}")

dataset = PoseRefinePairH5Dataset(cfg= cfg, h5_file='', mode='test')
# model = RefineNet(cfg= cfg, c_in= cfg['c_in']).cuda()

logging.info(f"Using pretrained model from {ckpt_dir}")
# ckpt = torch.load(ckpt_dir)
# if 'model' in ckpt:
#   ckpt = ckpt['model']
# model.load_state_dict(ckpt)


# model.cuda().eval()
logging.info("init done")
last_trans_update = None
last_rot_update = None

# for param in model.parameters():
#     param.requires_grad = False
##################################################3############# sunhan train
model_rgb = RefineNet_RGB(cfg= cfg, c_in=4).cuda()

# pretrained_weights = torch.load('/home/sunh/6D_ws/Fpose_rgb/weights/8.from_4080++/model_best_19.pth')['model_state_dict']   ### my weight
# pretrained_weights = torch.load(ckpt_dir)  ### ori weight
# for key in pretrained_weights.keys():
#   print(key)  # 或者使用其他逻辑以便找到正确的层
# 需要处理的权重
# with torch.no_grad():
#     pretrained_weights['encodeA.0.net.0.weight'] = pretrained_weights['encodeA.0.net.0.weight'][:,:4,:,:]
# 加载权重
# model_rgb.load_state_dict(pretrained_weights)


# # 冻结所有层的参数
# for param in  model_rgb.parameters():
#   param.requires_grad = False
#
# # 仅解冻 encodeA 部分
# for param in  model_rgb.encodeA.parameters():
#   param.requires_grad = True


parameters_to_train = list(filter(lambda p: p.requires_grad,  model_rgb.parameters()))
optimizer = optim.Adam(parameters_to_train, weight_decay=0.0, lr=0.0001)

##########################################################################################################################

from torch.utils.data import Dataset, DataLoader
from datasetloader import Dataset_Train,reset_object,get_tf_to_centered_mesh,make_crop_data_batch
batch_size = 64
epoches = 6400
train_data = Dataset_Train(batch_size=batch_size)
train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=False)


device = 'cuda:0'
iteration = 2
K = np.array([[567.53720406, 0.0, 312.66570357], [0.0, 569.36175922, 257.1729701], [0.0, 0.0, 1.0]])  ## mp6d
glctx = dr.RasterizeCudaContext(device)


# # 定义 warm-up 策略
# def warmup_lr_lambda(epoch):
#     if epoch < 5:  # 前 5 个 epoch 做 warm-up
#         return (epoch + 1) / 5  # 将学习率逐步增加
#     else:
#         return 1  # 之后保持目标学习率

# 使用 LambdaLR 调度器
# scheduler = LambdaLR(optimizer, lr_lambda=warmup_lr_lambda)
camera_intrinsics = torch.as_tensor(K, device='cuda', dtype=torch.float)
loss_fn = PoseReprojectionLoss2(camera_intrinsics,clamp_error=1.0)

##########################################################################################################################
for epoch in range(epoches):
    print('----------------------------------------------------- epoch:{} -----------------------------------------------------------------'.format(epoch))
    batch_number = 0
    for poses, rgb, depth, xyz_map, mesh_index, gt_poses in train_loader:
        poses = poses[0]
        rgb = rgb[0]
        depth = depth[0]
        xyz_map = xyz_map[0]

        mesh = train_data.meshes[mesh_index]
        model_center = train_data.model_centeres[mesh_index]
        mesh_tensors = train_data.mesh_tensorses[mesh_index]
        mesh_points = mesh_tensors['pos']
        if len(mesh_points) >= 2000:
            idxs = torch.randperm(len(mesh_points))[:2000]  # 随机排列并选择前 3000 个索引
        else:
            idxs1 = torch.arange(len(mesh_points))  # 前 len(mesh_points) 个索引
            idxs2 = torch.randint(0, len(mesh_points), (2000 - len(mesh_points),))  # 随机重复采样剩余的索引
            idxs = torch.cat([idxs1, idxs2], dim=0)  # 合并索引
        mesh_points = mesh_points[idxs]
        tf_to_center = train_data.tf_to_centeres[mesh_index]
        mesh_diameter = train_data.diameteres[mesh_index]
        to_origin = train_data.to_origines[mesh_index]
        extents = train_data.extentses[mesh_index]
        bbox_show = train_data.bbox_showes[mesh_index]

        ob_in_cams = poses
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        logging.info(f'ob_in_cams:{ob_in_cams.shape}')
        tf_to_center = np.eye(4)
        ob_centered_in_cams = ob_in_cams
        mesh_centered = mesh

        logging.info(f'  cfg.use_normal:{cfg.use_normal}')
        if not cfg.use_normal:
            normal_map = None

        crop_ratio = cfg['crop_ratio']
        logging.info(f"trans_normalizer:{cfg['trans_normalizer']}, rot_normalizer:{cfg['rot_normalizer']}")
        bs = 1024

        B_in_cams = torch.as_tensor(ob_centered_in_cams, device='cuda', dtype=torch.float)

        if mesh_tensors is None:
            mesh_tensors = make_mesh_tensors(mesh_centered)

        rgb_tensor = torch.as_tensor(rgb, device='cuda', dtype=torch.float)
        depth_tensor = torch.as_tensor(depth, device='cuda', dtype=torch.float)
        xyz_map_tensor = torch.as_tensor(xyz_map, device='cuda', dtype=torch.float)

        trans_normalizer = cfg['trans_normalizer']
        if not isinstance(trans_normalizer, float):
            trans_normalizer = torch.as_tensor(list(trans_normalizer), device='cuda', dtype=torch.float).reshape(1, 3)


        gt_poses = torch.as_tensor(gt_poses[0], device='cuda', dtype=torch.float)

        optimizer.zero_grad()

        for iteration_number in range(iteration):
            rot_residual, trans_residual = compute_residual(B_in_cams, gt_poses, cfg, mesh_diameter)
            logging.info("making cropped data")
            pose_data = make_crop_data_batch(cfg.input_resize, B_in_cams, mesh_centered, rgb_tensor, depth_tensor, K,
                                             crop_ratio=crop_ratio, normal_map=normal_map, xyz_map=xyz_map_tensor,
                                             cfg=cfg, glctx=glctx, mesh_tensors=mesh_tensors, dataset=dataset,
                                             mesh_diameter=mesh_diameter)
            B_in_cams = []
            for b in range(0, pose_data.rgbAs.shape[0], bs):
                A = torch.cat([pose_data.rgbAs[b:b + bs].cuda(), pose_data.xyz_mapAs[b:b + bs].cuda()], dim=1).float()
                B = torch.cat([pose_data.rgbBs[b:b + bs].cuda(), pose_data.xyz_mapBs[b:b + bs].cuda()], dim=1).float()
                logging.info("forward start")

                output_pre, x_rgb = model_rgb(A, B)


                trans_delta = output_pre["trans"]
                rot_mat_delta = torch.tanh(output_pre["rot"]) *    cfg['rot_normalizer'] #0.3490658503988659
                rot_mat_delta = so3_exp_map(rot_mat_delta).permute(0, 2, 1)

                trans_delta *= (mesh_diameter / 2)

                B_in_cam = egocentric_delta_pose_to_pose(pose_data.poseA[b:b + bs], trans_delta=trans_delta, ## pose_data.poseA[b:b + bs] ->  input pose
                                                         rot_mat_delta=rot_mat_delta)

                loss_rot = F.mse_loss(output_pre['rot'],rot_residual)
                loss_trans = F.mse_loss(output_pre['trans'],trans_residual)
                add_loss = loss_fn.compute_loss(mesh_points, B_in_cam, gt_poses) # add loss: ADD-(S)
                loss = add_loss *0.01 + loss_rot * 50 + 50 * loss_trans

                if iteration_number==0:
                    loss.backward(retain_graph=True)
                else:
                    loss.backward()

                print('epoch  {}   batch_number:{} iteration_number:{}   loss_all: {}   ROT: {}     TRAN: {} add_loss: {} '.format(
                    epoch, batch_number,iteration_number, loss,loss_rot * 50  , 50 * loss_trans,add_loss*0.1  ))

                B_in_cams.append(B_in_cam)
            B_in_cams = torch.cat(B_in_cams, dim=0).reshape(len(ob_in_cams), 4, 4)
        optimizer.step()

    batch_number = batch_number + 1

    # scheduler.step()

    if (epoch + 1) % 5 == 0:
        checkpoint = {
            'model_state_dict': model_rgb.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
        }
        # 保存权重到 model_best.pth
        torch.save(checkpoint, 'model_best2.pth')




















