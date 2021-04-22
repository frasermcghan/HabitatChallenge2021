import os

import numpy as np
import orbslam2
import torch
import quaternion

from habitat_baselines.slambased.mappers import DirectDepthMapper
from habitat_baselines.slambased.reprojection import (
    homogenize_p,
    project_tps_into_worldmap,
)
from habitat_baselines.slambased.utils import generate_2dgrid


class ORBSLAM2LocAgent:
    def __init__(self, config, follower, device=torch.device("cuda:0")):

        self.slam_vocab_path = config.SLAM_VOCAB_PATH
        assert os.path.isfile(self.slam_vocab_path)

        self.slam_settings_path = config.SLAM_SETTINGS_PATH
        assert os.path.isfile(self.slam_settings_path)

        self.slam = orbslam2.System(
            self.slam_vocab_path, self.slam_settings_path, orbslam2.Sensor.RGBD
        )

        self.slam.set_use_viewer(False)
        self.slam.initialize()
        self.device = device
        self.map_size_meters = config.MAP_SIZE
        self.map_cell_size = config.MAP_CELL_SIZE
        self.obstacle_th = config.MIN_PTS_IN_OBSTACLE
        self.depth_denorm = config.DEPTH_DENORM

        self.mapper = DirectDepthMapper(
            camera_height=config.CAMERA_HEIGHT,
            near_th=config.D_OBSTACLE_MIN,
            far_th=config.D_OBSTACLE_MAX,
            h_min=config.H_OBSTACLE_MIN,
            h_max=config.H_OBSTACLE_MAX,
            map_size=config.MAP_SIZE,
            map_cell_size=config.MAP_CELL_SIZE,
            device=device,
        )

        self.slam_to_world = 1.0
        self.timestep = 0.1
        self.timing = False
        self.follower = follower
        self.reset()
        return

    def reset(self):

        self.tracking_is_OK = False
        self.unseen_obstacle = False
        self.map2DObstacles = self.init_map2d()
        n, ch, height, width = self.map2DObstacles.size()
        self.coordinatesGrid = generate_2dgrid(height, width, False).to(self.device)
        self.pose6D = self.init_pose6d()
        self.pose6D_history = []
        self.position_history = []

        self.slam.reset()

        self.cur_time = 0
        self.toDoList = []

        if self.device != torch.device("cpu"):
            torch.cuda.empty_cache()

        return

    def update_internal_state(self, habitat_observation):

        rgb, depth = self.rgb_d_from_observation(habitat_observation)

        try:
            self.slam.process_image_rgbd(rgb, depth, self.cur_time)
            self.tracking_is_OK = str(self.slam.get_tracking_state()) == "OK"
        except BaseException:
            print("Warning!!!! ORBSLAM processing frame error")
            self.tracking_is_OK = False

        if self.tracking_is_OK:

            trajectory_history = np.array(self.slam.get_trajectory_points())

            self.pose6D = homogenize_p(
                torch.from_numpy(trajectory_history[-1])[1:].view(3, 4).to(self.device)
            ).view(1, 4, 4)

            self.trajectory_history = trajectory_history

            current_obstacles = self.mapper(
                torch.from_numpy(depth).to(self.device).squeeze(), self.pose6D
            ).to(self.device)

            self.current_obstacles = current_obstacles

            self.map2DObstacles = torch.max(
                self.map2DObstacles, current_obstacles.unsqueeze(0).unsqueeze(0),
            )
            return True

        else:
            return False

    def act(self, habitat_observation, goal_position):
        # Update internal state
        cc = 0

        update_is_ok = self.update_internal_state(habitat_observation)

        while not update_is_ok:
            update_is_ok = self.update_internal_state(habitat_observation)
            cc += 1
            if cc > 5:
                break

        current_pose = self.pose6D.detach().cpu().numpy().reshape(4, 4)
        self.position_history.append(current_pose)

        current_position = current_pose[0:3, -1]

        current_rotation = current_pose[0:3, 0:3]
        current_quat = quaternion.from_rotation_matrix(current_rotation)
        current_heading = quaternion.as_euler_angles(current_quat)[1]

        current_map = self.map2DObstacles.detach().cpu().numpy()

        best_action = self.follower.get_next_action(goal_pos=goal_position)

        return (current_position, current_heading, current_map, best_action)

    def init_pose6d(self):
        return torch.eye(4).float().to(self.device)

    def map_size_in_cells(self):
        return int(self.map_size_meters / self.map_cell_size)

    def init_map2d(self):
        return (
            torch.zeros(1, 1, self.map_size_in_cells(), self.map_size_in_cells())
            .float()
            .to(self.device)
        )

    def get_orientation_on_map(self):
        self.pose6D = self.pose6D.view(1, 4, 4)
        return torch.tensor(
            [
                [self.pose6D[0, 0, 0], self.pose6D[0, 0, 2]],
                [self.pose6D[0, 2, 0], self.pose6D[0, 2, 2]],
            ]
        )

    def get_position_on_map(self, pose=None, do_floor=True):
        if pose is not None:
            position = project_tps_into_worldmap(
                pose.view(1, 4, 4), self.map_cell_size, self.map_size_meters, do_floor,
            )
        else:
            position = project_tps_into_worldmap(
                self.pose6D.view(1, 4, 4),
                self.map_cell_size,
                self.map_size_meters,
                do_floor,
            )
        return position

    def rgb_d_from_observation(self, habitat_observation):
        rgb = habitat_observation["rgb"]
        depth = None
        if "depth" in habitat_observation:
            depth = self.depth_denorm * habitat_observation["depth"]
        return rgb, depth
