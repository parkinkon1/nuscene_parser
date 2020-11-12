import torch
from torch.utils.data import DataLoader
from dataset.nuscenes import NuscenesDataset, nuscenes_collate

import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: {}".format(device))


def load_CMU():
    sampling_rate = 2
    use_scene = True
    scene_size = (64, 64)
    ploss_type = 'map'

    test_partition = 'val'
    map_version = '2.0'
    sample_stride = 1
    multi_agent = 1
    num_workers = 20
    test_cache = "../intelpro_trajectory/data/nuscenes_val_cache.pkl"
    batch_size = 64

    dataset = NuscenesDataset(
        test_partition, map_version=map_version, sampling_rate=sampling_rate,
        sample_stride=sample_stride, use_scene=use_scene, scene_size=scene_size,
        ploss_type=ploss_type, num_workers=num_workers,
        cache_file=test_cache, multi_agent=multi_agent)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                             pin_memory=True, collate_fn=lambda x: nuscenes_collate(x), num_workers=1)

    print(f'Test Examples: {len(dataset)}')

    return dataset, data_loader


def load_pky():
    filename = 'nuscene_0.02_0.5.pickle'

    dataset = None
    with open(filename, 'rb') as f:
        dataset = pickle.load(f)

    batch_size = 64

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                             pin_memory=True, collate_fn=lambda x: nuscenes_collate(x), num_workers=1)

    print(f'Test Examples: {len(dataset)}')

    return dataset, data_loader


from torchvision import transforms
import torch.nn.functional as F
import cv2
from rasterization_q10.generator_dev import NusLoaderQ10
import numpy as np


def angle_diff(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0
    au = a / np.linalg.norm(a)
    bu = b / np.linalg.norm(b)
    return np.arccos(np.clip(np.dot(au, bu), -1.0, 1.0))


def calculateCurve(points):
    if len(points) < 3:
        return 0
    return abs(angle_diff(points[1] - points[0], points[-1] - points[0]))


def get_agent_mask(agent_past, agent_future, agent_translation, map_width=50, map_height=50):
    num_agents = len(agent_past)
    future_agent_masks = [True] * num_agents

    past_agents_traj = [[[0., 0.]] * 4] * num_agents
    future_agents_traj = [[[0., 0.]] * 6] * num_agents

    past_agents_traj = np.array(past_agents_traj)
    future_agents_traj = np.array(future_agents_traj)

    past_agents_traj_len = [4] * num_agents
    future_agents_traj_len = [6] * num_agents

    decode_start_vel = [[0., 0.]] * num_agents
    decode_start_pos = [[0., 0.]] * num_agents

    for idx, path in enumerate(zip(agent_past, agent_future)):
        past = path[0]
        future = path[1]
        pose = agent_translation[idx]

        # agent filtering
        side_length = map_width // 2
        if len(past) == 0 or len(future) == 0 \
                or np.max(pose) > side_length or np.min(pose) < -side_length:
            future_agent_masks[idx] = False

        # agent trajectory
        if len(past) < 4:
            past_agents_traj_len[idx] = len(past)
        for i, point in enumerate(past[:4]):
            past_agents_traj[idx, i] = point

        if len(future) < 6:
            future_agents_traj_len[idx] = len(future)
        for i, point in enumerate(future[:6]):
            future_agents_traj[idx, i] = point

        # vel, pose
        if len(future) != 0:
            decode_start_vel[idx] = (future[0] - agent_translation[idx]) / 0.5
        decode_start_pos[idx] = agent_translation[idx]

    return past_agents_traj, past_agents_traj_len, future_agents_traj, future_agents_traj_len, \
        future_agent_masks, decode_start_vel, decode_start_pos


class CustomLoader:
    def __init__(self):
        self.DATAROOT = '/datasets/nuscene/v1.0-mini'
        self.sampling_time = 3
        self.agent_time = 0  # zero for static mask, non-zero for overlap
        self.layer_names = ['drivable_area', 'road_segment', 'road_block',
                            'lane', 'ped_crossing', 'walkway', 'stop_line',
                            'carpark_area', 'road_divider', 'lane_divider']
        self.colors = [(255, 255, 255), (100, 255, 255), (255, 100, 255),
                       (255, 255, 100), (100, 100, 255), (100, 255, 100), (255, 100, 100),
                       (100, 100, 100), (50, 100, 50), (200, 50, 50), ]
        self.dataset = NusLoaderQ10(
            root=self.DATAROOT,
            sampling_time=self.sampling_time,
            agent_time=self.agent_time,
            layer_names=self.layer_names,
            colors=self.colors,
            resolution=0.1,
            meters_ahead=25,
            meters_behind=25,
            meters_left=25,
            meters_right=25)
        print("num_samples: {}".format(len(self.dataset)))

        self.p_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([23.0582], [27.3226]),
            transforms.Lambda(lambda x: F.log_softmax(x.reshape(-1), dim=0).reshape(x.shape[1:]))
        ])
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([23.0582], [27.3226])
        ])

        # init
        self.map_masks, self.map_img, self.agent_mask, self.xy_local, \
        self.virtual_mask, self.virtual_xy_local, self.idx = self.dataset[0]
        self.agent_past = self.xy_local[0]
        self.agent_future = self.xy_local[1]
        self.agent_translation = self.xy_local[2]
        self.virtual_past = self.virtual_xy_local[0]
        self.virtual_future = self.virtual_xy_local[1]
        self.virtual_translation = self.virtual_xy_local[2]

    def generateDistanceMaskFromColorMap(self, src, scene_size=(64, 64)):
        img = cv2.resize(src, scene_size)
        raw_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(raw_image, 100, 255, cv2.THRESH_BINARY)
        thresh = cv2.bitwise_not(thresh)
        raw_image = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)

        raw_map_image = cv2.resize(raw_image.astype(np.float32), dsize=(100, 100), interpolation=cv2.INTER_LINEAR)
        raw_map_image[raw_map_image < 0] = 0  # Uniform on drivable area
        raw_map_image = raw_map_image.max() - raw_map_image

        image = self.img_transform(raw_image)
        prior = self.p_transform(raw_map_image)

        return image, prior

    def load_data(self, idx, thres_min=None, thres_max=None):
        self.dataset.thres_min = thres_min
        self.dataset.thres_max = thres_max

        self.map_masks, self.map_img, self.agent_mask, self.xy_local, \
            self.virtual_mask, self.virtual_xy_local, self.idx = self.dataset[idx]

        self.agent_past = self.xy_local[0]
        self.agent_future = self.xy_local[1]
        self.agent_translation = self.xy_local[2]
        self.virtual_past = self.virtual_xy_local[0]
        self.virtual_future = self.virtual_xy_local[1]
        self.virtual_translation = self.virtual_xy_local[2]

    def dataProcessing(self, virtual=False):
        scene_id = self.idx
        # map mask & prior mask
        map_image, prior = self.generateDistanceMaskFromColorMap(self.map_masks[0], scene_size=(64, 64))

        if virtual:
            past_agents_traj, past_agents_traj_len, future_agents_traj, future_agents_traj_len, \
                future_agent_masks, decode_start_vel, decode_start_pos = \
                get_agent_mask(self.virtual_past, self.virtual_future, self.virtual_translation)
        else:
            past_agents_traj, past_agents_traj_len, future_agents_traj, future_agents_traj_len, \
                future_agent_masks, decode_start_vel, decode_start_pos = \
                get_agent_mask(self.agent_past, self.agent_future, self.agent_translation)

        episode = [past_agents_traj, past_agents_traj_len, future_agents_traj,
                   future_agents_traj_len, future_agent_masks,
                   np.array(decode_start_vel), np.array(decode_start_pos),
                   map_image, prior, scene_id]

        return episode

    def dataGeneration(self, thres=0.02, curved_ratio=0.3):
        episodes = []

        num_linear = 0
        num_curved = 0

        N = len(self.dataset)

        # count the number of curved agents
        for idx in range(N):
            self.load_data(idx, -1, thres)
            num_linear += len(self.agent_past)
            self.load_data(idx, thres, 99999)
            num_curved += len(self.agent_past)

        # original data
        for idx in range(N):
            self.load_data(idx, -1, 99999)
            episode = self.dataProcessing()
            if sum(episode[4]) > 0:
                episodes.append(episode)

        # generate curved data
        curved_target = (num_linear / (1 - curved_ratio)) * curved_ratio - num_curved
        index = 0
        while curved_target > 0:
            self.load_data(index, thres, 99999)
            episode = self.dataProcessing(virtual=True)

            n = sum(episode[4])

            if n > 0:
                episodes.append(episode)
                curved_target -= n

            index += 1
            if index > N - 1:
                index = 0

        return episodes
