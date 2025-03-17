# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import cv2
import numpy as np
from Utils import *

from estimater import *
from datareader import *
import argparse
import scipy.io as scio
import random


def reset_object(mesh=None):
    max_xyz = mesh.vertices.max(axis=0)
    min_xyz = mesh.vertices.min(axis=0)
    model_center = (min_xyz + max_xyz) / 2
    if mesh is not None:
        mesh_ori = mesh.copy()
        mesh = mesh.copy()
        mesh.vertices = mesh.vertices - model_center.reshape(1, 3)

    return mesh,model_center



def guess_translation(depth, mask, K):
    vs, us = np.where(mask > 0)
    if len(us) == 0:
        logging.info(f'mask is all zero')
        return np.zeros((3))
    uc = (us.min() + us.max()) / 2.0
    vc = (vs.min() + vs.max()) / 2.0
    valid = mask.astype(bool) & (depth >= 0.1)
    if not valid.any():
        logging.info(f"valid is empty")
        return np.zeros((3))

    zc = np.median(depth[valid])
    center = (np.linalg.inv(K) @ np.asarray([uc, vc, 1]).reshape(3, 1)) * zc


    return center.reshape(3)

def random_transform(M):
    rot_random = 20
    trans_random = 0.02
    # 随机平移
    delta_translation = np.random.uniform(-trans_random, trans_random, 3)  # -5cm to 5cm
    # 随机旋转
    delta_angles = np.random.uniform(np.deg2rad(-rot_random), np.deg2rad(rot_random), 3)
    # 创建平移矩阵
    T = np.eye(4)
    T[:3, 3] = delta_translation
    # 创建旋转矩阵（绕 Z, Y, X）
    R_z = np.array([[np.cos(delta_angles[2]), -np.sin(delta_angles[2]), 0],
                    [np.sin(delta_angles[2]), np.cos(delta_angles[2]), 0],
                    [0, 0, 1]])
    R_y = np.array([[np.cos(delta_angles[1]), 0, np.sin(delta_angles[1])],
                    [0, 1, 0],
                    [-np.sin(delta_angles[1]), 0, np.cos(delta_angles[1])]])
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(delta_angles[0]), -np.sin(delta_angles[0])],
                    [0, np.sin(delta_angles[0]), np.cos(delta_angles[0])]])
    R = R_z @ R_y @ R_x  # 组合旋转
    # 扩展旋转矩阵为4x4
    R_full = np.eye(4)
    R_full[:3, :3] = R
    # 组合变换
    M_prime = M @ R_full @ T
    return M_prime

def get_tf_to_centered_mesh(model_center):
    tf_to_center = torch.eye(4, dtype=torch.float, device='cuda')
    tf_to_center[:3,3] = -torch.as_tensor(model_center, device='cuda', dtype=torch.float)
    return tf_to_center


device = 'cuda:0'
iteration = 5
refiner = PoseRefinePredictor()
refiner.model.to(device)

mesh = trimesh.load('/home/sunh/6D_ws/ActivePose/FoundationPose-main/demo_data/sim/mesh/obj_000016.obj')
glctx = dr.RasterizeCudaContext(device)

mesh,model_center = reset_object(mesh)
mesh_tensors = make_mesh_tensors(mesh)
for k in mesh_tensors:
    mesh_tensors[k] = mesh_tensors[k].to(device)

tf_to_center = get_tf_to_centered_mesh(model_center)
diameter = compute_mesh_diameter(model_pts=mesh.vertices, n_sample=10000)

to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
bbox_show = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

############### mp6d  ###################
name = 16
data_path = '/home/sunh/6D_ws/other_network/megapose6d/local_data/examples/mp6d'
i = 0
K = np.array([[567.53720406, 0.0, 312.66570357], [0.0, 569.36175922, 257.1729701], [0.0, 0.0, 1.0]])  ## mp6d

R_error_rgbd_all = 0
R_error_rgb_all = 0
for obj_i in range(1,100):

    #### get GT pose
    obj_name = 'obj_{:02d}'.format(name - 15)
    with open(str(data_path) + f"/data/{i:04d}/{obj_i:06d}-box.txt") as f:
        bbox = f.readlines()
        obj_name_list = []
        for idx in range(len(bbox)):
            obj_bb = bbox[idx].split(" ")
            obj_name_list.append(obj_bb[0])
        if obj_name in obj_name_list:
            obj_id = obj_name_list.index(obj_name)
            obj_bb = bbox[obj_id].split(" ")
        else:
            print('no this object !!!')
    ### get intrinsic_matrix/pose/bbox
    dataFile = data_path + '/data/{:04d}/{:06d}-meta.mat'.format(i, obj_i)
    data = scio.loadmat(dataFile)
    poses = data['poses']
    gt_pose = poses[:, :, obj_id]
    poses_tmp = np.identity(4)
    gt_pose[:3, 3] = gt_pose[:3, 3] * 0.001
    poses_tmp[:3, :4] = gt_pose
    poses = poses_tmp.copy()



    ## read rgb and depth mask
    rgb = cv2.imread('./demo_data/0000/{:06d}-color.png'.format(obj_i))
    color = rgb.copy()
    depth = cv2.imread('./demo_data/0000/{:06d}-depth.png'.format(obj_i),
                       -1) / 1000.
    mask_test = cv2.imread(data_path + '/data/{:04d}/{:06d}-label.png'.format(0, obj_i), cv2.IMREAD_ANYDEPTH)
    mask_test[mask_test != (name-15)] = 0
    poses[:3,3] = guess_translation(depth, mask_test,K)  ## Attention!!!!!!!!!!!!!!!!!!!!!!    guess_translation
    poses = random_transform(poses)                      ## Attention!!!!!!!!!!!!!!!!!!!!!!    give noise to the GT Pose to get the initial Pose


    poses = np.array([poses])
    xyz_map = depth2xyzmap(depth, K)

    #### our rgb refiner
    poses_my, vis_rgb = refiner.predict_rgb(mesh=mesh, mesh_tensors=mesh_tensors, rgb=rgb, depth=depth, K=K,
                                 ob_in_cams=poses, normal_map=None, xyz_map=xyz_map,
                                 glctx=glctx, mesh_diameter=diameter, iteration=iteration,
                                 get_vis=1)
    #### FoundationPose refiner
    poses, vis_refine = refiner.predict(mesh=mesh, mesh_tensors=mesh_tensors, rgb=rgb, depth=depth, K=K,
                                 ob_in_cams=poses, normal_map=None, xyz_map=xyz_map,
                                 glctx=glctx, mesh_diameter=diameter, iteration=iteration,
                                 get_vis=3)

    # imageio.imwrite(f'vis_refiner.png', vis_refine)

    # R_error_rgbd = re(poses[0].data.cpu().numpy()[:3,:3],gt_pose[:3,:3])
    # R_error_rgb = re(poses_my[0].data.cpu().numpy()[:3,:3],gt_pose[:3,:3])
    #
    # t_error_rgbd = te(poses[0].data.cpu().numpy()[:3,3],gt_pose[:3,3])
    # t_error_rgb = te(poses_my[0].data.cpu().numpy()[:3,3],gt_pose[:3,3])
    #
    # R_error_rgbd_all = R_error_rgbd + R_error_rgbd_all
    # R_error_rgb_all = R_error_rgb + R_error_rgb_all
    #
    # print("R ERROR RGBD  {}, rgb  {}: ".format(R_error_rgbd, R_error_rgb) )
    # print("T ERROR RGBD  {}, rgb  {}: ".format(t_error_rgbd, t_error_rgb) )

    center_pose = poses_my[0].data.cpu().numpy()#@np.linalg.inv(to_origin)
    vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox_show)
    vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)
    cv2.imshow('1', vis[...,::-1])
    cv2.imshow('vis_rgb', vis_rgb)
    cv2.imshow('vis_refine', vis_refine)
    cv2.waitKey(0)

print("R_error_rgbd_all: ",R_error_rgbd_all/290.)
print("R_error_rgb_all: ",R_error_rgb_all/290.)

