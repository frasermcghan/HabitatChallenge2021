import math

import numpy as np
import torch

import habitat
from habitat_baselines.config.default import get_config


def rotate_origin_only(xy, radians):
    """Only rotate a point around the origin (0, 0)."""
    x, y = xy
    xx = x * math.cos(radians) + y * math.sin(radians)
    yy = -x * math.sin(radians) + y * math.cos(radians)

    return xx, yy


def homog_pose(xz, R):
    xyz = np.array([xz[0], xz[1], xz[1]])
    pose = np.eye(4)
    pose[0:3, -1] = xyz
    pose[0:3, 0:3] = R
    return torch.tensor(pose).unsqueeze(-1).float()


def habitat_to_orbslam(habitat_position, habitat_heading, init_position, init_heading):

    x = habitat_position[0] - init_position[0]
    z = -(habitat_position[2] - init_position[2])

    transformed_position = [x, z]
    transformed_position = rotate_origin_only(
        transformed_position, radians=init_heading
    )

    transformed_rotation = np.abs(init_heading) - np.abs(habitat_heading)

    return transformed_position, transformed_rotation


def save_trajectory(
    fig,
    ax,
    writer,
    rgb,
    depth,
    global_map,
    local_map,
    map_true_coords,
    map_slam_coords,
    map_start_coords,
    map_goal_coords,
):
    map_ax = ax[0, 0]
    obstacles_ax = ax[0, 1]
    rgb_ax = ax[1, 0]
    depth_ax = ax[1, 1]

    global_y_min = np.min(np.where(global_map.any(axis=1))[0])
    global_y_max = np.max(np.where(global_map.any(axis=1))[0])
    global_x_min = np.min(np.where(global_map.any(axis=0))[0])
    global_x_max = np.max(np.where(global_map.any(axis=0))[0])

    local_y_min = np.min(np.where(local_map.any(axis=1))[0])
    local_y_max = np.max(np.where(local_map.any(axis=1))[0])
    local_x_min = np.min(np.where(local_map.any(axis=0))[0])
    local_x_max = np.max(np.where(local_map.any(axis=0))[0])

    map_ax.imshow(global_map, cmap="Greys")

    obstacles_ax.imshow(local_map, cmap="Greys")

    rgb_ax.imshow(rgb)

    depth_ax.imshow(depth, cmap="Greys")

    map_ax.plot(
        [p[0, 1] for p in map_true_coords],
        [p[0, 0] for p in map_true_coords],
        color="red",
        label="True",
        linewidth=0.5,
    )

    map_ax.plot(
        [p[0, 1] for p in map_slam_coords],
        [p[0, 0] for p in map_slam_coords],
        color="blue",
        label="SLAM",
        linewidth=0.5,
    )

    map_ax.scatter(
        map_start_coords[0, 1],
        map_start_coords[0, 0],
        color="black",
        label="Start",
        s=1,
    )

    map_ax.scatter(
        map_goal_coords[0, 1],
        map_goal_coords[0, 0],
        color="green",
        label="Start",
        s=1,
        marker="*",
    )

    map_ax.set_title("Trajectory")
    map_ax.set_xticks([])
    map_ax.set_yticks([])
    map_ax.set_xlim(global_x_min - 20, global_x_max + 20)
    map_ax.set_ylim(global_y_min - 20, global_y_max + 20)

    obstacles_ax.set_title("Local Obstacle Map")
    obstacles_ax.set_xticks([])
    obstacles_ax.set_yticks([])
    obstacles_ax.set_xlim(local_x_min - 20, local_x_max + 20)
    obstacles_ax.set_ylim(local_y_min - 20, local_y_max + 20)

    rgb_ax.set_title("RGB")
    rgb_ax.set_xticks([])
    rgb_ax.set_yticks([])

    depth_ax.set_title("Depth")
    depth_ax.set_xticks([])
    depth_ax.set_yticks([])

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    writer.append_data(image)

    map_ax.clear()
    obstacles_ax.clear()
    rgb_ax.clear()
    depth_ax.clear()


def make_orb_config():
    baseline_config = get_config("configs/tasks/pointnav_rgbd.yaml")
    config = habitat.get_config("configs/tasks/pointnav_rgbd.yaml")

    config.defrost()
    baseline_config.defrost()

    config.TASK_CONFIG = baseline_config.TASK_CONFIG
    config.ORBSLAM2 = baseline_config.ORBSLAM2
    config.ORBSLAM2.SLAM_VOCAB_PATH = "data/ORBvoc.txt"
    config.ORBSLAM2.SLAM_SETTINGS_PATH = "configs/orbslam2/mp3d3_small1k.yaml"
    config.ORBSLAM2.DIST_TO_STOP = 0.2
    config.ORBSLAM2.MAP_CELL_SIZE = 0.10
    config.SIMULATOR.AGENT_0.SENSORS = [
        "RGB_SENSOR",
        "DEPTH_SENSOR",
    ]
    config.SIMULATOR.RGB_SENSOR.WIDTH = 256
    config.SIMULATOR.RGB_SENSOR.HEIGHT = 256
    config.SIMULATOR.DEPTH_SENSOR.WIDTH = 256
    config.SIMULATOR.DEPTH_SENSOR.HEIGHT = 256
    config.ORBSLAM2.CAMERA_HEIGHT = config.SIMULATOR.DEPTH_SENSOR.POSITION[1]
    config.ORBSLAM2.H_OBSTACLE_MIN = 0.3 * config.ORBSLAM2.CAMERA_HEIGHT
    config.ORBSLAM2.H_OBSTACLE_MAX = 1.0 * config.ORBSLAM2.CAMERA_HEIGHT
    config.ORBSLAM2.MIN_PTS_IN_OBSTACLE = config.SIMULATOR.DEPTH_SENSOR.WIDTH / 2.0

    config.freeze()
    baseline_config.freeze()

    return config
