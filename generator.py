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


class NusLoaderQ10(Dataset):
    def __init__(self, root='/datasets/nuscene/v1.0-mini', sampling_time=3, layer_names=None, colors=None):
        if layer_names is None:
            layer_names = ['drivable_area', 'road_segment', 'road_block',
                           'lane', 'ped_crossing', 'walkway', 'stop_line',
                           'carpark_area', 'road_divider', 'lane_divider']
        if colors is None:
            colors = [(255, 255, 255), (255, 255, 255), (255, 255, 255),
                      (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255),
                      (255, 255, 255), (255, 255, 255), (255, 255, 255),]
        self.root = root
        self.nus = NuScenes('v1.0-mini', dataroot=self.root)
        self.scenes = self.nus.scene
        self.samples = self.nus.sample

        self.layer_names = layer_names
        self.colors = colors

        self.helper = PredictHelper(self.nus)

        self.seconds = sampling_time

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        # 1. past_agents_traj : (Num obv agents in batch_i X 20 X 2)
        # 2. past_agents_traj_len : (Num obv agents in batch_i, )
        # 3. future_agents_traj : (Num pred agents in batch_i X 20 X 2)
        # 4. future_agents_traj_len : (Num pred agents in batch_i, )
        # 5. future_agent_masks : (Num obv agents in batch_i)
        # 6. decode_rel_pos: (Num pred agents in batch_i X 2)
        # 7. decode_start_pos: (Num pred agents in batch_i X 2)
        # 8. map_image : (3 X 224 X 224)
        # 9. scene ID: (string)

        sample = self.samples[idx]
        sample_token = sample['token']

        # 1. calculate ego pose
        sample_data_lidar = self.nus.get('sample_data', sample['data']['LIDAR_TOP'])
        ego_pose = self.nus.get('ego_pose', sample_data_lidar['ego_pose_token'])
        ego_pose_xy = ego_pose['translation']
        ego_pose_rotation = ego_pose['rotation']
        # 타임스탬프
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

        # 3. Generate Agent Trajectory
        annotation_tokens = sample['anns']
        num_agent = len(annotation_tokens)
        agents = []
        for ans_token in annotation_tokens:
            agent_states = []
            agent = self.nus.get('sample_annotation', ans_token)
            instance_token = agent['instance_token']

            # 에이전트 주행경로
            xy_global = agent['translation']
            past_xy_global = self.helper.get_past_for_agent(
                instance_token, sample_token, seconds=self.seconds, in_agent_frame=False)
            future_xy_global = self.helper.get_future_for_agent(
                instance_token, sample_token, seconds=self.seconds, in_agent_frame=False)

            # 로컬 주행경로
            xy_local = convert_global_coords_to_local(np.array([xy_global]), ego_pose_xy, ego_pose_rotation)
            past_xy_local = convert_global_coords_to_local(past_xy_global, ego_pose_xy, ego_pose_rotation)
            future_xy_local = convert_global_coords_to_local(future_xy_global, ego_pose_xy, ego_pose_rotation)

            # 에이전트 주행상태
            rot = agent['rotation']
            vel = self.helper.get_velocity_for_agent(instance_token, sample_token)
            accel = self.helper.get_acceleration_for_agent(instance_token, sample_token)
            yaw_rate = self.helper.get_heading_change_rate_for_agent(instance_token, sample_token)

            agent_states = {'present_pos': xy_global, 'past_pos': past_xy_global, 'future_pos': future_xy_global,
                            'rot': rot, 'vel': vel, 'accel': accel, 'yaw_rate': yaw_rate,
                            'present_local_xy': xy_local, 'past_local_xy': past_xy_local,
                            'future_local_xy': future_xy_local}

            agents.append(agent_states)

        return map_masks, agents

