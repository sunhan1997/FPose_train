import glob
import configparser
import os
import trimesh
import cv2
import numpy as np
import scipy.io as scio
from Utils import *
from learning.datasets.h5_dataset import *
from datareader import *
import random
import trimesh
from Utils import nvdiffrast_render
import imgaug.augmenters as iaa


import functools
def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator

def reset_object(mesh=None):
    max_xyz = mesh.vertices.max(axis=0)
    min_xyz = mesh.vertices.min(axis=0)
    model_center = (min_xyz + max_xyz) / 2
    if mesh is not None:
        mesh_ori = mesh.copy()
        mesh = mesh.copy()
        mesh.vertices = mesh.vertices - model_center.reshape(1, 3)

    return mesh,model_center


def get_tf_to_centered_mesh(model_center):
    tf_to_center = torch.eye(4, dtype=torch.float, device='cuda')
    tf_to_center[:3,3] = -torch.as_tensor(model_center, device='cuda', dtype=torch.float)
    return tf_to_center

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


def random_transform(random_rot=15,random_trans=0.0005,  M=None):
    # 随机平移
    # delta_translation = np.random.uniform(-random_trans, random_trans, 3)  # -5cm to 5cm
    xy_range = (-0.01, 0.01)
    z_range = (-0.05, 0.05)

    # 生成在指定范围内的随机向量
    delta_translation = np.array([
        np.random.uniform(*xy_range),  # x 和 y 在范围 [-0.02, 0.02]
        np.random.uniform(*xy_range),
        np.random.uniform(*z_range)  # z 在范围 [-0.05, 0.05]
    ])    # 随机旋转
    delta_angles = np.random.uniform(np.deg2rad(-random_rot), np.deg2rad(random_rot), 3)
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


def get_pose(random_rot=180):
    z_range = (0.5,0.9)
    z_translation = np.random.uniform(z_range[0], z_range[1])
    if z_translation<0.75:
        x_range = (-0.12, 0.12)
        y_range = (-0.06, 0.06)
    else:
        x_range = (-0.25, 0.25)
        y_range = (-0.17, 0.17)
    x_translation = np.random.uniform(x_range[0], x_range[1])
    y_translation = np.random.uniform(y_range[0], y_range[1])
    translation =  np.array([x_translation, y_translation, z_translation])

    # 随机旋转
    angles = np.random.uniform(np.deg2rad(-random_rot), np.deg2rad(random_rot), 3)
    # 创建平移矩阵
    # 创建旋转矩阵（绕 Z, Y, X）
    R_z = np.array([[np.cos(    angles[2]), -np.sin(    angles[2]), 0],
                    [np.sin(    angles[2]), np.cos(    angles[2]), 0],
                    [0, 0, 1]])
    R_y = np.array([[np.cos(    angles[1]), 0, np.sin(    angles[1])],
                    [0, 1, 0],
                    [-np.sin(    angles[1]), 0, np.cos(    angles[1])]])
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(    angles[0]), -np.sin(    angles[0])],
                    [0, np.sin(    angles[0]), np.cos(    angles[0])]])
    R = R_z @ R_y @ R_x  # 组合旋转

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = translation
    return T


def make_crop_data_batch(render_size, ob_in_cams, mesh, rgb, depth, K, crop_ratio, xyz_map, normal_map=None, mesh_diameter=None, cfg=None, glctx=None, mesh_tensors=None, dataset:PoseRefinePairH5Dataset=None):
  logging.info("Welcome make_crop_data_batch")
  H,W = depth.shape[:2]
  args = []
  method = 'box_3d'
  tf_to_crops = compute_crop_window_tf_batch(pts=mesh.vertices, H=H, W=W, poses=ob_in_cams, K=K, crop_ratio=crop_ratio, out_size=(render_size[1], render_size[0]), method=method, mesh_diameter=mesh_diameter)

  logging.info("make tf_to_crops done")

  B = len(ob_in_cams)
  poseA = torch.as_tensor(ob_in_cams, dtype=torch.float, device='cuda')

  bs = 512
  rgb_rs = []
  depth_rs = []
  normal_rs = []
  xyz_map_rs = []

  bbox2d_crop = torch.as_tensor(np.array([0, 0, cfg['input_resize'][0]-1, cfg['input_resize'][1]-1]).reshape(2,2), device='cuda', dtype=torch.float)
  bbox2d_ori = transform_pts(bbox2d_crop, tf_to_crops.inverse()).reshape(-1,4)

  start  = time.time()
  for b in range(0,len(poseA),bs):
    extra = {}
    rgb_r, depth_r, normal_r = nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=poseA[b:b+bs], context='cuda', get_normal=cfg['use_normal'], glctx=glctx, mesh_tensors=mesh_tensors, output_size=cfg['input_resize'], bbox2d=bbox2d_ori[b:b+bs], use_light=True, extra=extra)
    rgb_rs.append(rgb_r)
    depth_rs.append(depth_r[...,None])
    normal_rs.append(normal_r)
    xyz_map_rs.append(extra['xyz_map'])
  rgb_rs = torch.cat(rgb_rs, dim=0).permute(0,3,1,2) * 255
  depth_rs = torch.cat(depth_rs, dim=0).permute(0,3,1,2)  #(B,1,H,W)
  xyz_map_rs = torch.cat(xyz_map_rs, dim=0).permute(0,3,1,2)  #(B,3,H,W)
  Ks = torch.as_tensor(K, device='cuda', dtype=torch.float).reshape(1,3,3)
  if cfg['use_normal']:
    normal_rs = torch.cat(normal_rs, dim=0).permute(0,3,1,2)  #(B,3,H,W)

  logging.info("render done")


  ### sunhan   test
  # rgbBs = kornia.geometry.transform.warp_perspective(torch.as_tensor(rgb, dtype=torch.float, device='cuda').permute(2,0,1)[None].expand(B,-1,-1,-1), tf_to_crops, dsize=render_size, mode='bilinear', align_corners=False)
  ### sunhan   train
  rgbBs = kornia.geometry.transform.warp_perspective(rgb.permute(0,3,1,2), tf_to_crops, dsize=render_size, mode='bilinear', align_corners=False)
  if rgb_rs.shape[-2:]!=cfg['input_resize']:
    rgbAs = kornia.geometry.transform.warp_perspective(rgb_rs, tf_to_crops, dsize=render_size, mode='bilinear', align_corners=False)
  else:
    rgbAs = rgb_rs
  if xyz_map_rs.shape[-2:]!=cfg['input_resize']:
    xyz_mapAs = kornia.geometry.transform.warp_perspective(xyz_map_rs, tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)
  else:
    xyz_mapAs = xyz_map_rs

  ### sunhan   test
  # xyz_mapBs = kornia.geometry.transform.warp_perspective(torch.as_tensor(xyz_map, device='cuda', dtype=torch.float).permute(2,0,1)[None].expand(B,-1,-1,-1), tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)  #(B,3,H,W)
  ### sunhan  train
  xyz_mapBs = kornia.geometry.transform.warp_perspective(xyz_map.permute(0,3,1,2), tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)  #(B,3,H,W)

  if cfg['use_normal']:
    normalAs = kornia.geometry.transform.warp_perspective(normal_rs, tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)
    normalBs = kornia.geometry.transform.warp_perspective(torch.as_tensor(normal_map, dtype=torch.float, device='cuda').permute(2,0,1)[None].expand(B,-1,-1,-1), tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)
  else:
    normalAs = None
    normalBs = None

  logging.info("warp done")

  mesh_diameters = torch.ones((len(rgbAs)), dtype=torch.float, device='cuda')*mesh_diameter
  pose_data = BatchPoseData(rgbAs=rgbAs, rgbBs=rgbBs, depthAs=None, depthBs=None, normalAs=normalAs, normalBs=normalBs, poseA=poseA, poseB=None, xyz_mapAs=xyz_mapAs, xyz_mapBs=xyz_mapBs, tf_to_crops=tf_to_crops, Ks=Ks, mesh_diameters=mesh_diameters)
  pose_data = dataset.transform_batch(batch=pose_data, H_ori=H, W_ori=W, bound=1)

  logging.info("pose batch data done")

  return pose_data



class Dataset_Train(object):

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.num_mesh = 1  ### the number of meshes
        self.meshes = []
        self.model_centeres = []
        self.mesh_tensorses = []
        self.tf_to_centeres = []
        self.diameteres = []
        self.to_origines = []
        self.extentses = []
        self.bbox_showes = []
        for i in range(self.num_mesh):
        #
            print('mesh index: ', i)
            # mesh = trimesh.load('/home/sunh/6D_ws/Fpose_rgb/train_data/ori_mesh/obj_{:06d}.ply'.format(i))
            mesh = trimesh.load('./train_data/lmo_mp6d/obj_{:06d}.ply'.format(i))
            mesh, model_center = reset_object(mesh)
            mesh_tensor = make_mesh_tensors(mesh)
            for k in mesh_tensor:
                mesh_tensor[k] = mesh_tensor[k]

            tf_to_center = get_tf_to_centered_mesh(model_center)
            diameter = compute_mesh_diameter(model_pts=mesh.vertices, n_sample=10000)

            to_origin, extents = trimesh.bounds.oriented_bounds(    mesh)
            bbox_show = np.stack([-    extents / 2,     extents / 2], axis=0).reshape(2, 3)

            self.meshes.append(mesh)
            self.model_centeres.append(model_center)
            self.mesh_tensorses.append(mesh_tensor)
            self.tf_to_centeres.append(tf_to_center)
            self.diameteres.append(diameter)
            self.to_origines.append(to_origin)
            self.extentses.append(extents)
            self.bbox_showes.append(bbox_show)

        self.K = np.array([[567.53720406, 0.0, 312.66570357], [0.0, 569.36175922, 257.1729701], [0.0, 0.0, 1.0]])  #

        background_dir = '/home/sunh/6D_ws/CT_GDR/datasets/VOCdevkit/VOC2012/JPEGImages'  ## background
        # 获取背景图像的路径列表
        self.background_images = [os.path.join(background_dir, img) for img in os.listdir(background_dir)
                                  if img.endswith(('.png', '.jpg', '.jpeg'))]
        if not self.background_images:
            raise ValueError("No background images found in the provided directory.")

        # 定义图像增强流水线
        self.augmenter = iaa.Sequential([
            iaa.Sometimes(0.3, iaa.CoarseDropout( p=0.2, size_percent=0.05)),
            iaa.Sometimes(0.3, iaa.GaussianBlur((0, 2.0))),
            iaa.Sometimes(0.4, iaa.Add((-25, 25), per_channel=0.3)),
            iaa.Sometimes(0.3, iaa.Invert(0.2, per_channel=True))
        ])



    def add_random_light(self, image):
        brightness = np.random.randint(-100, 100)
        image_with_light = cv2.add(image, brightness)
        return image_with_light


    def __len__(self):
        self.num_image = self.num_mesh * 500
        return self.num_image

    ## 把数据放入网络
    def __getitem__(self, index):


        mesh_index = index % self.num_mesh
        mesh = self.meshes[mesh_index]

        H = 480
        W = 640

        ob_in_cams=[]
        for i in range(self.batch_size ):
            ob_in_cam = get_pose()
            ob_in_cams.append(ob_in_cam)

        ob_in_cams = np.array(ob_in_cams)
        gt_poses = ob_in_cams.copy()
        ob_in_cams = torch.from_numpy(ob_in_cams).cuda().to(torch.float32)


        rgb_r, depth_r, normal_r = nvdiffrast_render(K=self.K, H=H, W=W, ob_in_cams=ob_in_cams, context='cuda',
                                                     get_normal=False, glctx=None, mesh_tensors=None, mesh=mesh,
                                                     output_size=None, bbox2d=None,
                                                     use_light=True, extra={})

        rgbs = rgb_r.cpu().numpy()
        depths = depth_r.cpu().numpy()

        xy_offset = 0.008 #np.random.uniform(0.000, 0.01)
        rot_offset = 20 #np.random.uniform(0, 15)

        in_poses = []
        xyz_maps = []
        bg_rgbs = []
        for i in range(self.batch_size ):
            gt_pose = gt_poses[i]
            in_pose = random_transform(random_rot=rot_offset, random_trans=xy_offset, M=gt_pose)
            in_poses.append(in_pose)

            depth = depths[i]
            xyz_map = depth2xyzmap(depth, self.K)
            xyz_maps.append(xyz_map)

            background_path = random.choice(self.background_images)
            background_image = cv2.imread(background_path)
            bg_rgb = cv2.resize(background_image,(640,480))
            rgb = rgbs[i]*255
            bg_rgb[rgb>0] = rgb[rgb>0]
            # bg_rgb = self.augmenter(image=bg_rgb)
            # bg_rgb = self.add_random_light(bg_rgb)
            bg_rgbs.append(bg_rgb)

            # cv2.imshow('bg_rgb', bg_rgb)
            # cv2.imshow('rgb', rgb)
            # cv2.imshow('xyz_map', xyz_map)
            # cv2.waitKey()


        in_poses = np.array(in_poses)
        bg_rgbs = np.array(bg_rgbs)
        xyz_maps = np.array(xyz_maps)
        return (in_poses,bg_rgbs,depths,xyz_maps, mesh_index,gt_poses )



# from torch.utils.data import Dataset, DataLoader
# batch_size = 1
# epoches = 500
# train_data = Dataset_Train(24)
#
# train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
#
# for poses, rgb, depth, xyz_map, mesh_index in train_loader:
#     rgb = rgb[0]
#     depth = depth[0]
#     xyz_map = xyz_map[0]
#
#     mesh =  train_data.meshes[mesh_index]
#     model_center =  train_data.model_centeres[mesh_index]
#     mesh_tensors =  train_data.mesh_tensorses[mesh_index]
#     tf_to_center =  train_data.tf_to_centeres[mesh_index]
#     diameter =  train_data.diameteres[mesh_index]
#     to_origin =  train_data.to_origines[mesh_index]
#     extents =  train_data.extentses[mesh_index]
#     bbox_show =  train_data.bbox_showes[mesh_index]
#
#
#     a = poses
#     b = rgb
#     c = depth
#     d = xyz_map
#     pass
#
