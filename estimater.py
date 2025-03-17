# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import time

import numpy as np
import torch

from Utils import *
from datareader import *
import itertools
from learning.training.predict_score import *
from learning.training.predict_pose_refine2 import * # sunhan add
# from learning.training.predict_pose_refine2 import *
import yaml


class FoundationPose:
  def __init__(self, model_pts, model_normals, symmetry_tfs=None, mesh=None, scorer:ScorePredictor=None, refiner:PoseRefinePredictor=None, glctx=None, debug=0, debug_dir='/home/bowen/debug/novel_pose_debug/'):
    self.gt_pose = None
    self.ignore_normal_flip = True
    self.debug = debug
    self.debug_dir = debug_dir
    os.makedirs(debug_dir, exist_ok=True)

    self.reset_object(model_pts, model_normals, symmetry_tfs=symmetry_tfs, mesh=mesh)
    self.make_rotation_grid(min_n_views=40, inplane_step=60)

    self.glctx = glctx

    if scorer is not None:
      self.scorer = scorer
    else:
      self.scorer = ScorePredictor()

    if refiner is not None:
      self.refiner = refiner
    else:
      self.refiner = PoseRefinePredictor()

    self.pose_last = None   # Used for tracking; per the centered mesh


  def reset_object(self, model_pts, model_normals, symmetry_tfs=None, mesh=None):
    max_xyz = mesh.vertices.max(axis=0)
    min_xyz = mesh.vertices.min(axis=0)
    self.model_center = (min_xyz+max_xyz)/2
    if mesh is not None:
      self.mesh_ori = mesh.copy()
      mesh = mesh.copy()
      mesh.vertices = mesh.vertices - self.model_center.reshape(1,3)

    model_pts = mesh.vertices
    self.diameter = compute_mesh_diameter(model_pts=mesh.vertices, n_sample=10000)
    self.vox_size = max(self.diameter/20.0, 0.003)
    logging.info(f'self.diameter:{self.diameter}, vox_size:{self.vox_size}')
    self.dist_bin = self.vox_size/2
    self.angle_bin = 20  # Deg
    pcd = toOpen3dCloud(model_pts, normals=model_normals)
    pcd = pcd.voxel_down_sample(self.vox_size)
    self.max_xyz = np.asarray(pcd.points).max(axis=0)
    self.min_xyz = np.asarray(pcd.points).min(axis=0)
    self.pts = torch.tensor(np.asarray(pcd.points), dtype=torch.float32, device='cuda')
    self.normals = F.normalize(torch.tensor(np.asarray(pcd.normals), dtype=torch.float32, device='cuda'), dim=-1)
    logging.info(f'self.pts:{self.pts.shape}')
    self.mesh_path = None
    self.mesh = mesh
    if self.mesh is not None:
      self.mesh_path = f'/tmp/{uuid.uuid4()}.obj'
      self.mesh.export(self.mesh_path)
    self.mesh_tensors = make_mesh_tensors(self.mesh)

    if symmetry_tfs is None:
      self.symmetry_tfs = torch.eye(4).float().cuda()[None]
    else:
      self.symmetry_tfs = torch.as_tensor(symmetry_tfs, device='cuda', dtype=torch.float)

    logging.info("reset done")



  def get_tf_to_centered_mesh(self):
    tf_to_center = torch.eye(4, dtype=torch.float, device='cuda')
    tf_to_center[:3,3] = -torch.as_tensor(self.model_center, device='cuda', dtype=torch.float)
    return tf_to_center


  def to_device(self, s='cuda:0'):
    for k in self.__dict__:
      self.__dict__[k] = self.__dict__[k]
      if torch.is_tensor(self.__dict__[k]) or isinstance(self.__dict__[k], nn.Module):
        logging.info(f"Moving {k} to device {s}")
        self.__dict__[k] = self.__dict__[k].to(s)
    for k in self.mesh_tensors:
      logging.info(f"Moving {k} to device {s}")
      self.mesh_tensors[k] = self.mesh_tensors[k].to(s)
    if self.refiner is not None:
      self.refiner.model.to(s)
    if self.scorer is not None:
      self.scorer.model.to(s)
    if self.glctx is not None:
      self.glctx = dr.RasterizeCudaContext(s)



  def make_rotation_grid(self, min_n_views=40, inplane_step=60):
    cam_in_obs = sample_views_icosphere(n_views=min_n_views)
    logging.info(f'cam_in_obs:{cam_in_obs.shape}')
    rot_grid = []
    for i in range(len(cam_in_obs)):
      for inplane_rot in np.deg2rad(np.arange(0, 360, inplane_step)):
        cam_in_ob = cam_in_obs[i]
        R_inplane = euler_matrix(0,0,inplane_rot)
        cam_in_ob = cam_in_ob@R_inplane
        ob_in_cam = np.linalg.inv(cam_in_ob)
        rot_grid.append(ob_in_cam)

    rot_grid = np.asarray(rot_grid)
    logging.info(f"rot_grid:{rot_grid.shape}")
    rot_grid = mycpp.cluster_poses(30, 99999, rot_grid, self.symmetry_tfs.data.cpu().numpy())
    rot_grid = np.asarray(rot_grid)
    logging.info(f"after cluster, rot_grid:{rot_grid.shape}")
    self.rot_grid = torch.as_tensor(rot_grid, device='cuda', dtype=torch.float)
    logging.info(f"self.rot_grid: {self.rot_grid.shape}")


  def generate_random_pose_hypo(self, K, rgb, depth, mask, scene_pts=None):
    '''
    @scene_pts: torch tensor (N,3)
    '''
    ob_in_cams = self.rot_grid.clone()
    center = self.guess_translation(depth=depth, mask=mask, K=K)
    ob_in_cams[:,:3,3] = torch.tensor(center, device='cuda', dtype=torch.float).reshape(1,3)
    return ob_in_cams

  def rotation_matrix_similarity(self, matrix1, matrix2):
    # 计算两个旋转矩阵的 Frobenius 范数
    frobenius_norm = np.linalg.norm(matrix1 - matrix2, 'fro')
    # 返回范数作为相似度度量
    return frobenius_norm

  def generate_random_pose_hypo_rgb(self, K, rgb, depth, mask, scene_pts=None, guess_pose=None):
    '''
    @scene_pts: torch tensor (N,3)
    '''
    ob_in_cams = self.rot_grid.clone()
    center = self.guess_translation_rgb(depth=depth, mask=mask, K=K, pred_z = guess_pose[2,3])
    # center = guess_pose[:3,3]
    ob_in_cams[:,:3,3] = torch.tensor(center, device='cuda', dtype=torch.float).reshape(1,3)
    return ob_in_cams


  def generate_random_pose_hypo_rgb_less(self, K, rgb, depth, mask, scene_pts=None, guess_pose=None):
    '''
    @scene_pts: torch tensor (N,3)
    '''
    ob_in_cams = self.rot_grid.clone()
    center = self.guess_translation_rgb(depth=depth, mask=mask, K=K, pred_z = guess_pose[2,3])
    scores = []
    for i in range(len(ob_in_cams)):
      score = self.rotation_matrix_similarity(ob_in_cams[i,:3,:3].data.cpu().numpy(),guess_pose[:3,:3])
      scores.append((score, i))  # 保存分数及其索引，以便找到对应的矩阵
    # 按分数从低到高排序，取出前10个最相似的矩阵索引
    top_10_scores = sorted(scores, key=lambda x: x[0])[:3]
    # 获取前10个最相似旋转矩阵的索引
    top_10_indices = [idx for _, idx in top_10_scores]
    ob_in_cams = ob_in_cams[top_10_indices]  # 提取最相似的旋转矩阵

    ob_in_cams = torch.tensor(np.array([guess_pose]), device='cuda', dtype=torch.float)
    # center = self.guess_translation(depth=depth, mask=mask, K=K)
    # ob_in_cams[:,:3,3] = torch.tensor(center, device='cuda', dtype=torch.float).reshape(1,3)

    return ob_in_cams

  def generate_random_pose_hypo_rgb_less2(self, K, rgb, depth, mask, scene_pts=None, guess_pose=None):
    '''
    @scene_pts: torch tensor (N,3)
    '''
    ob_in_cams = self.rot_grid.clone()
    center = self.guess_translation_rgb(depth=depth, mask=mask, K=K, pred_z=guess_pose[2, 3])
    scores = []
    for i in range(len(ob_in_cams)):
      score = self.rotation_matrix_similarity(ob_in_cams[i, :3, :3].data.cpu().numpy(), guess_pose[:3, :3])
      scores.append((score, i))  # 保存分数及其索引，以便找到对应的矩阵
    # 按分数从低到高排序，取出前10个最相似的矩阵索引
    top_10_scores = sorted(scores, key=lambda x: x[0])[:3]
    # 获取前10个最相似旋转矩阵的索引
    top_10_indices = [idx for _, idx in top_10_scores]
    ob_in_cams = ob_in_cams[top_10_indices]  # 提取最相似的旋转矩阵

    # ob_in_cams = torch.tensor(np.array([guess_pose]), device='cuda', dtype=torch.float)
    # center = self.guess_translation(depth=depth, mask=mask, K=K)
    ob_in_cams[:, :3, 3] = torch.tensor(center, device='cuda', dtype=torch.float).reshape(1, 3)

    return ob_in_cams

    # # 原始的 ob_in_cams 是 (10, 4, 4) 的张量
    # ob_in_cams_forward = ob_in_cams.clone()  # 向前平移 0.01m
    # ob_in_cams_backward = ob_in_cams.clone()  # 向后平移 0.01m
    # # 修改 z 轴平移量
    # ob_in_cams_forward[:, 2, 3] += 0.008  # 向前增加 0.01m
    # ob_in_cams_backward[:, 2, 3] -= 0.08  # 向后减少 0.01m
    # # 将原始、向前平移和向后平移的张量组合在一起
    # combined_ob_in_cams = torch.cat((ob_in_cams, ob_in_cams_forward, ob_in_cams_backward), dim=0)
    #
    # return combined_ob_in_cams


  def guess_translation(self, depth, mask, K):
    vs,us = np.where(mask>0)
    if len(us)==0:
      logging.info(f'mask is all zero')
      return np.zeros((3))
    uc = (us.min()+us.max())/2.0
    vc = (vs.min()+vs.max())/2.0
    valid = mask.astype(bool) & (depth>=0.1)
    if not valid.any():
      logging.info(f"valid is empty")
      return np.zeros((3))

    zc = np.median(depth[valid])
    center = (np.linalg.inv(K)@np.asarray([uc,vc,1]).reshape(3,1))*zc

    if self.debug>=2:
      pcd = toOpen3dCloud(center.reshape(1,3))
      o3d.io.write_point_cloud(f'{self.debug_dir}/init_center.ply', pcd)

    return center.reshape(3)

  def guess_translation_rgb(self, depth, mask, K,pred_z):
    vs,us = np.where(mask>0)
    if len(us)==0:
      logging.info(f'mask is all zero')
      return np.zeros((3))
    uc = (us.min()+us.max())/2.0
    vc = (vs.min()+vs.max())/2.0
    valid = mask.astype(bool) & (depth>=0.1)
    if not valid.any():
      logging.info(f"valid is empty")
      return np.zeros((3))

    zc = pred_z
    center = (np.linalg.inv(K)@np.asarray([uc,vc,1]).reshape(3,1))*zc

    if self.debug>=2:
      pcd = toOpen3dCloud(center.reshape(1,3))
      o3d.io.write_point_cloud(f'{self.debug_dir}/init_center.ply', pcd)

    return center.reshape(3)

  def register(self, K, rgb, depth, ob_mask, ob_id=None, glctx=None, iteration=5):
    '''Copmute pose from given pts to self.pcd
    @pts: (N,3) np array, downsampled scene points
    '''
    set_seed(0)
    logging.info('Welcome')

    if self.glctx is None:
      if glctx is None:
        self.glctx = dr.RasterizeCudaContext()
        # self.glctx = dr.RasterizeGLContext()
      else:
        self.glctx = glctx

    depth = erode_depth(depth, radius=2, device='cuda')
    depth = bilateral_filter_depth(depth, radius=2, device='cuda')

    if self.debug>=2:
      xyz_map = depth2xyzmap(depth, K)
      valid = xyz_map[...,2]>=0.1
      pcd = toOpen3dCloud(xyz_map[valid], rgb[valid])
      o3d.io.write_point_cloud(f'{self.debug_dir}/scene_raw.ply',pcd)
      cv2.imwrite(f'{self.debug_dir}/ob_mask.png', (ob_mask*255.0).clip(0,255))

    normal_map = None
    valid = (depth>=0.1) & (ob_mask>0)
    if valid.sum()<4:
      logging.info(f'valid too small, return')
      pose = np.eye(4)
      pose[:3,3] = self.guess_translation(depth=depth, mask=ob_mask, K=K)
      return pose

    if self.debug>=2:
      imageio.imwrite(f'{self.debug_dir}/color.png', rgb)
      cv2.imwrite(f'{self.debug_dir}/depth.png', (depth*1000).astype(np.uint16))
      valid = xyz_map[...,2]>=0.1
      pcd = toOpen3dCloud(xyz_map[valid], rgb[valid])
      o3d.io.write_point_cloud(f'{self.debug_dir}/scene_complete.ply',pcd)

    self.H, self.W = depth.shape[:2]
    self.K = K
    self.ob_id = ob_id
    self.ob_mask = ob_mask

    poses = self.generate_random_pose_hypo(K=K, rgb=rgb, depth=depth, mask=ob_mask, scene_pts=None)
    poses = poses.data.cpu().numpy()
    logging.info(f'poses:{poses.shape}')
    center = self.guess_translation(depth=depth, mask=ob_mask, K=K)

    poses = torch.as_tensor(poses, device='cuda', dtype=torch.float)
    poses[:,:3,3] = torch.as_tensor(center.reshape(1,3), device='cuda')

    add_errs = self.compute_add_err_to_gt_pose(poses)
    logging.info(f"after viewpoint, add_errs min:{add_errs.min()}")


    xyz_map = depth2xyzmap(depth, K)

    start_refine = time.time()
    ### sunhan
    poses, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb, depth=depth, K=K, ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map, xyz_map=xyz_map, glctx=self.glctx, mesh_diameter=self.diameter, iteration=iteration, get_vis=self.debug>=2)
    # poses, vis = self.refiner.predict_rgb(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb, depth=depth, K=K, ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map, xyz_map=xyz_map, glctx=self.glctx, mesh_diameter=self.diameter, iteration=iteration, get_vis=self.debug>=2)
    end_refine = time.time()
    print("refine :  ", end_refine - start_refine)
    if vis is not None:
      imageio.imwrite(f'{self.debug_dir}/vis_refiner.png', vis)

    start_score = time.time()
    scores, vis = self.scorer.predict(mesh=self.mesh, rgb=rgb, depth=depth, K=K, ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map, mesh_tensors=self.mesh_tensors, glctx=self.glctx, mesh_diameter=self.diameter, get_vis=self.debug>=2)
    end_score = time.time()
    print("scorer :  ", end_score - start_score)

    if vis is not None:
      imageio.imwrite(f'{self.debug_dir}/vis_score.png', vis)

    start_add = time.time()
    add_errs = self.compute_add_err_to_gt_pose(poses)
    end_add = time.time()
    print("compute_add_err_to_gt_pose :  ", end_add - start_add)

    logging.info(f"final, add_errs min:{add_errs.min()}")

    ids = torch.as_tensor(scores).argsort(descending=True)
    logging.info(f'sort ids:{ids}')
    scores = scores[ids]
    poses = poses[ids]

    logging.info(f'sorted scores:{scores}')

    best_pose = poses[0]@self.get_tf_to_centered_mesh()
    self.pose_last = poses[0]
    self.best_id = ids[0]

    self.poses = poses
    self.scores = scores

    return best_pose.data.cpu().numpy()


  def register_rgb(self, K, rgb, depth, ob_mask, ob_id=None, glctx=None, iteration=5, guess_pose=None):
    '''Copmute pose from given pts to self.pcd
    @pts: (N,3) np array, downsampled scene points
    '''
    set_seed(0)
    logging.info('Welcome')

    if self.glctx is None:
      if glctx is None:
        self.glctx = dr.RasterizeCudaContext()
        # self.glctx = dr.RasterizeGLContext()
      else:
        self.glctx = glctx

    depth = erode_depth(depth, radius=2, device='cuda:0')
    depth = bilateral_filter_depth(depth, radius=2, device='cuda')

    if self.debug>=2:
      xyz_map = depth2xyzmap(depth, K)
      valid = xyz_map[...,2]>=0.1
      pcd = toOpen3dCloud(xyz_map[valid], rgb[valid])
      o3d.io.write_point_cloud(f'{self.debug_dir}/scene_raw.ply',pcd)
      cv2.imwrite(f'{self.debug_dir}/ob_mask.png', (ob_mask*255.0).clip(0,255))

    normal_map = None
    # valid = (depth>=0.1) & (ob_mask>0)
    # if valid.sum()<4:
    #   logging.info(f'valid too small, return')
    #   pose = np.eye(4)
    #   pose[:3,3] = self.guess_translation(depth=depth, mask=ob_mask, K=K)
    #   return pose

    if self.debug>=2:
      imageio.imwrite(f'{self.debug_dir}/color.png', rgb)
      cv2.imwrite(f'{self.debug_dir}/depth.png', (depth*1000).astype(np.uint16))
      valid = xyz_map[...,2]>=0.1
      pcd = toOpen3dCloud(xyz_map[valid], rgb[valid])
      o3d.io.write_point_cloud(f'{self.debug_dir}/scene_complete.ply',pcd)

    self.H, self.W = depth.shape[:2]
    self.K = K
    self.ob_id = ob_id
    self.ob_mask = ob_mask

    # poses = self.generate_random_pose_hypo_rgb(K=K, rgb=rgb, depth=depth, mask=ob_mask, scene_pts=None, guess_pose=guess_pose)  #sunhan
    poses = self.generate_random_pose_hypo_rgb_less(K=K, rgb=rgb, depth=depth, mask=ob_mask, scene_pts=None, guess_pose=guess_pose)
    poses = poses.data.cpu().numpy()
    logging.info(f'poses:{poses.shape}')
    # center = self.guess_translation(depth=depth, mask=ob_mask, K=K)

    poses = torch.as_tensor(poses, device='cuda', dtype=torch.float)
    # poses[:,:3,3] = torch.as_tensor(center.reshape(1,3), device='cuda')

    add_errs = self.compute_add_err_to_gt_pose(poses)
    logging.info(f"after viewpoint, add_errs min:{add_errs.min()}")



    xyz_map = depth2xyzmap(depth, K)

    start_refine = time.time()
    ### sunhan
    poses, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb, depth=depth, K=K, ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map, xyz_map=xyz_map, glctx=self.glctx, mesh_diameter=self.diameter, iteration=iteration, get_vis=self.debug>=2)
    end_refine = time.time()

    # print("refine :  ", end_refine - start_refine)
    # if vis is not None:
    #   imageio.imwrite(f'{self.debug_dir}/vis_refiner.png', vis)
    #   cv2.imwrite(f'{self.debug_dir}/vis_refiner.png', vis)
    #
    # start_score = time.time()
    # scores, vis = self.scorer.predict(mesh=self.mesh, rgb=rgb, depth=depth, K=K, ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map, mesh_tensors=self.mesh_tensors, glctx=self.glctx, mesh_diameter=self.diameter, get_vis=self.debug>=2)
    # end_score = time.time()
    # print("scorer :  ", end_score - start_score)
    #
    # if vis is not None:
    #   imageio.imwrite(f'{self.debug_dir}/vis_score.png', vis)
    #
    # start_add = time.time()
    # add_errs = self.compute_add_err_to_gt_pose(poses)
    # end_add = time.time()
    # print("compute_add_err_to_gt_pose :  ", end_add - start_add)
    #
    # logging.info(f"final, add_errs min:{add_errs.min()}")
    #
    # ids = torch.as_tensor(scores).argsort(descending=True)
    # logging.info(f'sort ids:{ids}')
    # scores = scores[ids]
    # poses = poses[ids]
    #
    # logging.info(f'sorted scores:{scores}')

    best_pose = poses[0]@self.get_tf_to_centered_mesh()
    self.pose_last = poses[0]
    # self.best_id = ids[0]

    # self.poses = poses
    # self.scores = scores

    return best_pose.data.cpu().numpy()

  def compute_add_err_to_gt_pose(self, poses):
    '''
    @poses: wrt. the centered mesh
    '''
    return -torch.ones(len(poses), device='cuda', dtype=torch.float)


  def track_one(self, rgb, depth, K, iteration, extra={}):
    if self.pose_last is None:
      logging.info("Please init pose by register first")
      raise RuntimeError
    logging.info("Welcome")

    depth = torch.as_tensor(depth, device='cuda', dtype=torch.float)
    depth = erode_depth(depth, radius=2, device='cuda:0')
    depth = bilateral_filter_depth(depth, radius=2, device='cuda')

    logging.info("depth processing done")

    xyz_map = depth2xyzmap_batch(depth[None], torch.as_tensor(K, dtype=torch.float, device='cuda')[None], zfar=np.inf)[0]



    # pose, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb, depth=depth, K=K, ob_in_cams=self.pose_last.reshape(1,4,4).data.cpu().numpy(), normal_map=None, xyz_map=xyz_map, mesh_diameter=self.diameter, glctx=self.glctx, iteration=iteration, get_vis=self.debug>=2)
    pose, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb, depth=depth, K=K, ob_in_cams=self.pose_last, normal_map=None, xyz_map=xyz_map, mesh_diameter=self.diameter, glctx=self.glctx, iteration=iteration, get_vis=self.debug>=2)
    logging.info("pose done")
    if self.debug>=2:
      extra['vis'] = vis
    self.pose_last = pose
    # return (pose@self.get_tf_to_centered_mesh()).data.cpu().numpy().reshape(4,4)
    return (pose@self.get_tf_to_centered_mesh()).data.cpu().numpy().reshape(4,4),vis

  def track_one_rgb(self, rgb, depth, K, iteration, extra={}):
    if self.pose_last is None:
      logging.info("Please init pose by register first")
      raise RuntimeError
    logging.info("Welcome")

    depth = torch.as_tensor(depth, device='cuda', dtype=torch.float)
    depth = erode_depth(depth, radius=2, device='cuda:0')
    depth = bilateral_filter_depth(depth, radius=2, device='cuda')

    logging.info("depth processing done")

    xyz_map = depth2xyzmap_batch(depth[None], torch.as_tensor(K, dtype=torch.float, device='cuda')[None], zfar=np.inf)[0]

    # pose, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb, depth=depth, K=K, ob_in_cams=self.pose_last.reshape(1,4,4).data.cpu().numpy(), normal_map=None, xyz_map=xyz_map, mesh_diameter=self.diameter, glctx=self.glctx, iteration=iteration, get_vis=self.debug>=2)
    pose, vis = self.refiner.predict_rgb(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb, depth=depth, K=K, ob_in_cams=self.pose_last, normal_map=None, xyz_map=xyz_map, mesh_diameter=self.diameter, glctx=self.glctx, iteration=iteration, get_vis=self.debug>=2)
    logging.info("pose done")
    if self.debug>=2:
      extra['vis'] = vis
    self.pose_last = pose
    # self.pose_last[0][2,3] = torch.Tensor([0.45])
    # return (pose@self.get_tf_to_centered_mesh()).data.cpu().numpy().reshape(4,4)
    return (pose@self.get_tf_to_centered_mesh()).data.cpu().numpy().reshape(4,4),vis

  ############################################################################  generate more  hypo  pose    ############################################################################

  def register_psoe_unc(self, K, rgb, depth, ob_mask, ob_id=None, glctx=None, iteration=5):
    '''Copmute pose from given pts to self.pcd
    @pts: (N,3) np array, downsampled scene points
    '''
    set_seed(0)
    logging.info('Welcome')

    if self.glctx is None:
      if glctx is None:
        self.glctx = dr.RasterizeCudaContext()
        # self.glctx = dr.RasterizeGLContext()
      else:
        self.glctx = glctx

    depth = erode_depth(depth, radius=2, device='cuda')
    depth = bilateral_filter_depth(depth, radius=2, device='cuda')

    if self.debug>=2:
      xyz_map = depth2xyzmap(depth, K)
      valid = xyz_map[...,2]>=0.1
      pcd = toOpen3dCloud(xyz_map[valid], rgb[valid])
      o3d.io.write_point_cloud(f'{self.debug_dir}/scene_raw.ply',pcd)
      cv2.imwrite(f'{self.debug_dir}/ob_mask.png', (ob_mask*255.0).clip(0,255))

    normal_map = None
    valid = (depth>=0.1) & (ob_mask>0)
    if valid.sum()<4:
      logging.info(f'valid too small, return')
      pose = np.eye(4)
      pose[:3,3] = self.guess_translation(depth=depth, mask=ob_mask, K=K)
      return pose

    if self.debug>=2:
      imageio.imwrite(f'{self.debug_dir}/color.png', rgb)
      cv2.imwrite(f'{self.debug_dir}/depth.png', (depth*1000).astype(np.uint16))
      valid = xyz_map[...,2]>=0.1
      pcd = toOpen3dCloud(xyz_map[valid], rgb[valid])
      o3d.io.write_point_cloud(f'{self.debug_dir}/scene_complete.ply',pcd)

    self.H, self.W = depth.shape[:2]
    self.K = K
    self.ob_id = ob_id
    self.ob_mask = ob_mask

    poses = self.generate_random_pose_hypo(K=K, rgb=rgb, depth=depth, mask=ob_mask, scene_pts=None)
    poses = poses.data.cpu().numpy()
    logging.info(f'poses:{poses.shape}')
    center = self.guess_translation(depth=depth, mask=ob_mask, K=K)

    poses = torch.as_tensor(poses, device='cuda', dtype=torch.float)
    poses[:,:3,3] = torch.as_tensor(center.reshape(1,3), device='cuda')

    add_errs = self.compute_add_err_to_gt_pose(poses)
    logging.info(f"after viewpoint, add_errs min:{add_errs.min()}")


    xyz_map = depth2xyzmap(depth, K)

    start_refine = time.time()
    poses, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb, depth=depth, K=K, ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map, xyz_map=xyz_map, glctx=self.glctx, mesh_diameter=self.diameter, iteration=iteration, get_vis=self.debug>=2)
    end_refine = time.time()
    print("refine :  ", end_refine - start_refine)
    if vis is not None:
      imageio.imwrite(f'{self.debug_dir}/vis_refiner.png', vis)

    start_score = time.time()
    scores, vis = self.scorer.predict(mesh=self.mesh, rgb=rgb, depth=depth, K=K, ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map, mesh_tensors=self.mesh_tensors, glctx=self.glctx, mesh_diameter=self.diameter, get_vis=self.debug>=2)
    end_score = time.time()
    print("scorer :  ", end_score - start_score)

    if vis is not None:
      imageio.imwrite(f'{self.debug_dir}/vis_score.png', vis)

    start_add = time.time()
    add_errs = self.compute_add_err_to_gt_pose(poses)
    end_add = time.time()
    print("compute_add_err_to_gt_pose :  ", end_add - start_add)

    logging.info(f"final, add_errs min:{add_errs.min()}")

    ids = torch.as_tensor(scores).argsort(descending=True)
    logging.info(f'sort ids:{ids}')
    scores = scores[ids]
    poses = poses[ids]

    logging.info(f'sorted scores:{scores}')

    best_pose = poses[0]@self.get_tf_to_centered_mesh()
    self.pose_last = poses[0]
    self.best_id = ids[0]

    self.poses = poses
    self.scores = scores

    best_pose_10 = []
    best_score_10 = []
    for i in range(10):
      best_pose = poses[i] @ self.get_tf_to_centered_mesh()
      best_pose_10.append(best_pose.data.cpu().numpy())
      best_score_10.append(scores[i].data.cpu().numpy())

    return best_pose_10, best_score_10
    # return poses.data.cpu().numpy(), scores.data.cpu().numpy()




  #######################################################  zero shot rgb refine ################################################################
  def register_psoe_rgb_refine(self, K, rgb, depth, ob_mask, ob_id=None, glctx=None, iteration=5):
    '''Copmute pose from given pts to self.pcd
    @pts: (N,3) np array, downsampled scene points
    '''
    set_seed(0)
    logging.info('Welcome')

    if self.glctx is None:
      if glctx is None:
        self.glctx = dr.RasterizeCudaContext()
        # self.glctx = dr.RasterizeGLContext()
      else:
        self.glctx = glctx

    depth = erode_depth(depth, radius=2, device='cuda')
    depth = bilateral_filter_depth(depth, radius=2, device='cuda')

    if self.debug >= 2:
      xyz_map = depth2xyzmap(depth, K)
      valid = xyz_map[..., 2] >= 0.1
      pcd = toOpen3dCloud(xyz_map[valid], rgb[valid])
      o3d.io.write_point_cloud(f'{self.debug_dir}/scene_raw.ply', pcd)
      cv2.imwrite(f'{self.debug_dir}/ob_mask.png', (ob_mask * 255.0).clip(0, 255))

    normal_map = None
    valid = (depth >= 0.1) & (ob_mask > 0)
    if valid.sum() < 4:
      logging.info(f'valid too small, return')
      pose = np.eye(4)
      pose[:3, 3] = self.guess_translation(depth=depth, mask=ob_mask, K=K)
      return pose

    if self.debug >= 2:
      imageio.imwrite(f'{self.debug_dir}/color.png', rgb)
      cv2.imwrite(f'{self.debug_dir}/depth.png', (depth * 1000).astype(np.uint16))
      valid = xyz_map[..., 2] >= 0.1
      pcd = toOpen3dCloud(xyz_map[valid], rgb[valid])
      o3d.io.write_point_cloud(f'{self.debug_dir}/scene_complete.ply', pcd)

    self.H, self.W = depth.shape[:2]
    self.K = K
    self.ob_id = ob_id
    self.ob_mask = ob_mask

    poses = self.generate_random_pose_hypo(K=K, rgb=rgb, depth=depth, mask=ob_mask, scene_pts=None)
    poses = poses.data.cpu().numpy()
    logging.info(f'poses:{poses.shape}')
    center = self.guess_translation(depth=depth, mask=ob_mask, K=K)

    poses = torch.as_tensor(poses, device='cuda', dtype=torch.float)
    poses[:, :3, 3] = torch.as_tensor(center.reshape(1, 3), device='cuda')

    add_errs = self.compute_add_err_to_gt_pose(poses)
    logging.info(f"after viewpoint, add_errs min:{add_errs.min()}")

    xyz_map = depth2xyzmap(depth, K)

    start_refine = time.time()
    poses, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb, depth=depth, K=K,
                                      ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map, xyz_map=xyz_map,
                                      glctx=self.glctx, mesh_diameter=self.diameter, iteration=iteration,
                                      get_vis=self.debug >= 2)
    end_refine = time.time()
    print("refine :  ", end_refine - start_refine)
    if vis is not None:
      imageio.imwrite(f'{self.debug_dir}/vis_refiner.png', vis)

    start_score = time.time()
    scores, vis = self.scorer.predict(mesh=self.mesh, rgb=rgb, depth=depth, K=K, ob_in_cams=poses.data.cpu().numpy(),
                                      normal_map=normal_map, mesh_tensors=self.mesh_tensors, glctx=self.glctx,
                                      mesh_diameter=self.diameter, get_vis=self.debug >= 2)
    end_score = time.time()
    print("scorer :  ", end_score - start_score)

    if vis is not None:
      imageio.imwrite(f'{self.debug_dir}/vis_score.png', vis)

    start_add = time.time()
    add_errs = self.compute_add_err_to_gt_pose(poses)
    end_add = time.time()
    print("compute_add_err_to_gt_pose :  ", end_add - start_add)

    logging.info(f"final, add_errs min:{add_errs.min()}")

    ids = torch.as_tensor(scores).argsort(descending=True)
    logging.info(f'sort ids:{ids}')
    scores = scores[ids]
    poses = poses[ids]

    logging.info(f'sorted scores:{scores}')

    best_pose = poses[0] @ self.get_tf_to_centered_mesh()
    self.pose_last = poses[0]
    self.best_id = ids[0]

    self.poses = poses
    self.scores = scores

    best_pose_10 = []
    best_score_10 = []
    for i in range(10):
      best_pose = poses[i] @ self.get_tf_to_centered_mesh()
      best_pose_10.append(best_pose.data.cpu().numpy())
      best_score_10.append(scores[i].data.cpu().numpy())

    return best_pose_10, best_score_10
    # return poses.data.cpu().numpy(), scores.data.cpu().numpy()






  def register_rgb2(self, K, rgb, depth, ob_mask, ob_id=None, glctx=None, iteration=5, guess_pose=None, net = None):

    '''Copmute pose from given pts to self.pcd
    @pts: (N,3) np array, downsampled scene points
    '''
    set_seed(0)
    logging.info('Welcome')

    if self.glctx is None:
      if glctx is None:
        self.glctx = dr.RasterizeCudaContext()
        # self.glctx = dr.RasterizeGLContext()
      else:
        self.glctx = glctx

    depth = erode_depth(depth, radius=2, device='cuda:0')
    depth = bilateral_filter_depth(depth, radius=2, device='cuda')

    if self.debug>=2:
      xyz_map = depth2xyzmap(depth, K)
      valid = xyz_map[...,2]>=0.1
      pcd = toOpen3dCloud(xyz_map[valid], rgb[valid])
      o3d.io.write_point_cloud(f'{self.debug_dir}/scene_raw.ply',pcd)
      cv2.imwrite(f'{self.debug_dir}/ob_mask.png', (ob_mask*255.0).clip(0,255))

    normal_map = None
    valid = (depth>=0.1) & (ob_mask>0)
    if valid.sum()<4:
      logging.info(f'valid too small, return')
      pose = np.eye(4)
      pose[:3,3] = self.guess_translation(depth=depth, mask=ob_mask, K=K)
      return pose

    if self.debug>=2:
      imageio.imwrite(f'{self.debug_dir}/color.png', rgb)
      cv2.imwrite(f'{self.debug_dir}/depth.png', (depth*1000).astype(np.uint16))
      valid = xyz_map[...,2]>=0.1
      pcd = toOpen3dCloud(xyz_map[valid], rgb[valid])
      o3d.io.write_point_cloud(f'{self.debug_dir}/scene_complete.ply',pcd)

    self.H, self.W = depth.shape[:2]
    self.K = K
    self.ob_id = ob_id
    self.ob_mask = ob_mask

    poses = self.generate_random_pose_hypo_rgb(K=K, rgb=rgb, depth=depth, mask=ob_mask, scene_pts=None, guess_pose=guess_pose)  #sunhan
    # poses = self.generate_random_pose_hypo_rgb_less(K=K, rgb=rgb, depth=depth, mask=ob_mask, scene_pts=None, guess_pose=guess_pose)
    poses = poses.data.cpu().numpy()
    logging.info(f'poses:{poses.shape}')
    # center = self.guess_translation(depth=depth, mask=ob_mask, K=K)

    poses = torch.as_tensor(poses, device='cuda', dtype=torch.float)
    # poses[:,:3,3] = torch.as_tensor(center.reshape(1,3), device='cuda')

    add_errs = self.compute_add_err_to_gt_pose(poses)
    logging.info(f"after viewpoint, add_errs min:{add_errs.min()}")



    xyz_map = depth2xyzmap(depth, K)

    start_refine = time.time()
    ### sunhan
    poses, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb, depth=depth, K=K, ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map, xyz_map=xyz_map, glctx=self.glctx, mesh_diameter=self.diameter, iteration=iteration, get_vis=self.debug>=2)
    end_refine = time.time()

    print("refine :  ", end_refine - start_refine)
    if vis is not None:
      imageio.imwrite(f'{self.debug_dir}/vis_refiner.png', vis)
      cv2.imwrite(f'{self.debug_dir}/vis_refiner.png', vis)

    # start_score = time.time()
    # scores, vis = self.scorer.predict(mesh=self.mesh, rgb=rgb, depth=depth, K=K, ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map, mesh_tensors=self.mesh_tensors, glctx=self.glctx, mesh_diameter=self.diameter, get_vis=self.debug>=2)
    # end_score = time.time()
    # print("scorer :  ", end_score - start_score)

    if vis is not None:
      imageio.imwrite(f'{self.debug_dir}/vis_score.png', vis)

    start_add = time.time()
    add_errs = self.compute_add_err_to_gt_pose(poses)
    end_add = time.time()
    print("compute_add_err_to_gt_pose :  ", end_add - start_add)

    logging.info(f"final, add_errs min:{add_errs.min()}")

    # ids = torch.as_tensor(scores).argsort(descending=True)
    # logging.info(f'sort ids:{ids}')
    # scores = scores[ids]
    # poses = poses[ids]

    # logging.info(f'sorted scores:{scores}')

    best_pose = poses[0]@self.get_tf_to_centered_mesh()
    # self.pose_last = poses[0]
    # self.best_id = ids[0]

    # self.poses = poses
    # self.scores = scores

    return best_pose.data.cpu().numpy()



