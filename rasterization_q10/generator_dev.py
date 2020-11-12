import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset

import os
from PIL import Image
import pickle

from nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils

from rasterization_q10.input_representation.static_layers import StaticLayerRasterizer
from rasterization_q10.input_representation.agents import AgentBoxesWithFadedHistory
from rasterization_q10.input_representation.interface import InputRepresentation
from rasterization_q10.input_representation.combinators import Rasterizer
from rasterization_q10 import PredictHelper

from pyquaternion import Quaternion
import numpy as np

from rasterization_q10.helper import convert_global_coords_to_local
import matplotlib.pyplot as plt


def curvature(A, B, C):
    # Augment Columns
    A_aug = np.append(A, 1)
    B_aug = np.append(B, 1)
    C_aug = np.append(C, 1)

    # Calculate Area of Triangle
    matrix = np.column_stack((A_aug, B_aug, C_aug))
    area = 1 / 2 * np.linalg.det(matrix)

    # Special case: Two or more points are equal
    if np.all(A == B) or np.all(B == C):
        curvature = 0
    else:
        curvature = 4 * area / (np.linalg.norm(A - B) * np.linalg.norm(B - C) * np.linalg.norm(C - A))

    # Return Menger curvature
    return curvature


def calculateCurve(points):
    if len(points) < 3:
        return 0
    curvature_list = np.empty(0)
    for i in range(len(points) - 2):
        A = points[i]
        B = points[i + 1]
        C = points[i + 2]
        curvature_value = abs(curvature(A, B, C))
        curvature_list = np.append(curvature_list, curvature_value)
    return np.average(curvature_list)


class NusLoaderQ10(Dataset):
    def __init__(self, root='/datasets/nuscene/v1.0-mini', sampling_time=3, agent_time=0, layer_names=None,
                 colors=None, resolution: float = 0.1,  # meters / pixel
                 meters_ahead: float = 25, meters_behind: float = 25,
                 meters_left: float = 25, meters_right: float = 25, version='v1.0-mini'):
        if layer_names is None:
            layer_names = ['drivable_area', 'road_segment', 'road_block',
                           'lane', 'ped_crossing', 'walkway', 'stop_line',
                           'carpark_area', 'road_divider', 'lane_divider']
        if colors is None:
            colors = [(255, 255, 255), (255, 255, 255), (255, 255, 255),
                      (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255),
                      (255, 255, 255), (255, 255, 255), (255, 255, 255), ]
        self.root = root
        self.nus = NuScenes(version, dataroot=self.root)
        self.scenes = self.nus.scene
        self.samples = self.nus.sample

        self.layer_names = layer_names
        self.colors = colors

        self.helper = PredictHelper(self.nus)

        self.seconds = sampling_time
        self.agent_seconds = agent_time

        self.static_layer = StaticLayerRasterizer(self.helper, layer_names=self.layer_names, colors=self.colors,
                                                  resolution=resolution, meters_ahead=meters_ahead,
                                                  meters_behind=meters_behind,
                                                  meters_left=meters_left, meters_right=meters_right)
        self.agent_layer = AgentBoxesWithFadedHistory(self.helper, seconds_of_history=self.agent_seconds,
                                                      resolution=resolution, meters_ahead=meters_ahead,
                                                      meters_behind=meters_behind,
                                                      meters_left=meters_left, meters_right=meters_right)

        self.thres_min = -1
        self.thres_max = 99999

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample_token = sample['token']

        # 1. calculate ego pose
        sample_data_lidar = self.nus.get('sample_data', sample['data']['LIDAR_TOP'])
        ego_pose = self.nus.get('ego_pose', sample_data_lidar['ego_pose_token'])
        ego_pose_xy = ego_pose['translation']
        ego_pose_rotation = ego_pose['rotation']
        # 타임스탬프
        timestamp = ego_pose['timestamp']

        # 2. Generate Map
        scene = self.nus.get('scene', sample['scene_token'])
        log = self.nus.get('log', scene['log_token'])
        location = log['location']
        nus_map = NuScenesMap(dataroot=self.root, map_name=location)

        map_masks, lanes, map_img = self.static_layer.generate_mask(ego_pose_xy, ego_pose_rotation, sample_token)

        # 3. Generate Agent Trajectory
        agent_mask, xy_global = self.agent_layer.generate_mask(
            ego_pose_xy, ego_pose_rotation, sample_token, self.seconds, self.thres_min, self.thres_max)

        xy_local = []
        for idx, global_xy in enumerate(xy_global):
            pose_xy = []
            if idx == 2:
                if len(global_xy) == 0:
                    xy_local.append(np.array([]))
                    continue
                # global_xy = global_xy
                xy_local.append(convert_global_coords_to_local(global_xy, ego_pose_xy, ego_pose_rotation))
                continue
            for i in range(len(global_xy)):
                if len(global_xy[i]) < 1:
                    pose_xy.append(np.array([]))
                else:
                    pose_xy.append(convert_global_coords_to_local(global_xy[i], ego_pose_xy, ego_pose_rotation))
            xy_local.append(pose_xy)
        # xy_local = np.array(xy_local)

        # 4. Generate Virtual Agent Trajectory
        lane_tokens = list(lanes.keys())
        num_lanes = len(lane_tokens)
        lanes_disc = [np.array(lanes[token])[:, :2] for token in lane_tokens]
        lanes_arc = np.empty(0)
        for seq in lanes_disc:
            lanes_arc = np.append(lanes_arc, calculateCurve(seq))

        virtual_mask, virtual_xy = self.agent_layer.generate_virtual_mask(
            ego_pose_xy, ego_pose_rotation, lanes_disc, sample_token, thres_min=self.thres_min, thres_max=self.thres_max)

        virtual_xy_local = []
        for idx, global_xy in enumerate(virtual_xy):
            pose_xy = []
            if idx == 2:
                if len(global_xy) == 0:
                    virtual_xy_local.append(np.array([]))
                    continue
                # global_xy = global_xy
                virtual_xy_local.append(convert_global_coords_to_local(global_xy, ego_pose_xy, ego_pose_rotation))
                continue
            for i in range(len(global_xy)):
                if len(global_xy[i]) < 1:
                    pose_xy.append(np.array([]))
                else:
                    pose_xy.append(convert_global_coords_to_local(global_xy[i], ego_pose_xy, ego_pose_rotation))
            virtual_xy_local.append(pose_xy)
        # virtual_xy_local = np.array(virtual_xy_local)

        return map_masks, map_img, agent_mask, xy_local, virtual_mask, virtual_xy_local, idx

    def render_sample(self, idx):
        sample = self.samples[idx]
        sample_token = sample['token']
        self.nus.render_sample(sample_token)

    def render_scene(self, idx):
        sample = self.samples[idx]
        sample_token = sample['token']
        scene = self.nus.get('scene', sample['scene_token'])
        log = self.nus.get('log', scene['log_token'])
        location = log['location']
        nus_map = NuScenesMap(dataroot=self.root, map_name=location)
        layer_names = ['road_segment', 'lane', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area']
        camera_channel = 'CAM_FRONT'
        nus_map.render_map_in_image(self.nus, sample_token, layer_names=layer_names, camera_channel=camera_channel)

    def render_map(self, idx, combined=True):
        sample = self.samples[idx]
        sample_token = sample['token']

        sample_data_lidar = self.nus.get('sample_data', sample['data']['LIDAR_TOP'])
        ego_pose = self.nus.get('ego_pose', sample_data_lidar['ego_pose_token'])
        ego_pose_xy = ego_pose['translation']
        ego_pose_rotation = ego_pose['rotation']
        timestamp = ego_pose['timestamp']

        # 2. Generate Map & Agent Masks
        scene = self.nus.get('scene', sample['scene_token'])
        log = self.nus.get('log', scene['log_token'])
        location = log['location']
        nus_map = NuScenesMap(dataroot=self.root, map_name=location)

        static_layer = StaticLayerRasterizer(self.helper, layer_names=self.layer_names, colors=self.colors)
        agent_layer = AgentBoxesWithFadedHistory(self.helper, seconds_of_history=self.seconds)

        map_masks, lanes, map_img = static_layer.generate_mask(ego_pose_xy, ego_pose_rotation, sample_token)
        agent_mask = agent_layer.generate_mask(ego_pose_xy, ego_pose_rotation, sample_token)

        if combined:
            plt.subplot(1, 2, 1)
            plt.title("combined map")
            plt.imshow(map_img)
            plt.subplot(1, 2, 2)
            plt.title("agent")
            plt.imshow(agent_mask)
            plt.show()
        else:
            num_labels = len(self.layer_names)
            num_rows = num_labels // 3
            fig, ax = plt.subplots(num_rows, 3, figsize=(10, 3 * num_rows))
            for row in range(num_rows):
                for col in range(3):
                    num = 3 * row + col
                    if num == num_labels - 1:
                        break
                    ax[row][col].set_title(self.layer_names[num])
                    ax[row][col].imshow(map_masks[num])
            plt.show()
