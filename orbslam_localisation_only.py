import imageio
import matplotlib.pyplot as plt
import torch

import habitat
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat_baselines.utils.env_utils import make_env_fn
from orbslam_localiser import ORBSLAM2LocAgent
from utils import (
    habitat_to_orbslam,
    homog_pose,
    make_orb_config,
    save_trajectory,
)

config = make_orb_config()

env_class = habitat.Env
env = make_env_fn(config, env_class)

follower = ShortestPathFollower(env.sim, goal_radius=0.05, return_one_hot=False)

localiser = ORBSLAM2LocAgent(config.ORBSLAM2, follower)

# n_episodes = env.number_of_episodes
n_episodes = 50
visualise = False

fig, ax = plt.subplots(2, 2, constrained_layout=True, figsize=(1280 // 96, 720 // 96))

for episode in range(n_episodes):

    habitat_positions = []
    habitat_headings = []

    slam_positions = []
    slam_headings = []

    map_true_coords = []
    map_slam_coords = []

    count_steps = 0

    observations = env.reset()
    localiser.reset()

    init_position = env.sim.get_agent_state().position
    init_heading = observations["heading"]

    map_start_coords = localiser.get_position_on_map(pose=torch.eye(4))

    goal_position = env.current_episode.goals[0].position
    goal_heading = 0.0
    goal_position_orb, goal_heading_orb = habitat_to_orbslam(
        goal_position, goal_heading, init_position, init_heading
    )
    goal_pose_orb = homog_pose(goal_position_orb, goal_heading_orb)

    map_goal_coords = localiser.get_position_on_map(pose=goal_pose_orb)

    scene_name = env.current_episode.scene_id.split(sep="/")[-1].strip(".glb")

    writer = imageio.get_writer(f"{scene_name}_{episode}.mp4", fps=5, quality=5)

    print(f"\nStarting episode {episode+1}/{n_episodes} in {scene_name}...")

    print("Stepping around...")
    while not env.episode_over:

        rgb = observations["rgb"]
        depth = observations["depth"]

        habitat_position = env.sim.get_agent_state().position
        habitat_heading = observations["heading"]

        transformed_position, transformed_heading = habitat_to_orbslam(
            habitat_position, habitat_heading, init_position, init_heading
        )

        habitat_positions.append(transformed_position)
        habitat_headings.append(transformed_heading)

        est_position, est_heading, est_map, best_action = localiser.act(
            observations, goal_position
        )

        slam_positions.append(est_position)
        slam_headings.append(est_heading)

        habitat_pose = homog_pose(transformed_position, transformed_heading)

        if visualise:

            slam_map_coords = localiser.get_position_on_map().detach().cpu().numpy()
            map_slam_coords.append(slam_map_coords)

            true_map_coords = localiser.get_position_on_map(pose=habitat_pose)
            map_true_coords.append(true_map_coords)

            global_map = localiser.map2DObstacles.detach().cpu().numpy().squeeze() > 0.5

            local_map = (
                localiser.current_obstacles.detach().cpu().numpy().squeeze() > 0.5
            )

            save_trajectory(
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
            )

        observations = env.step(best_action)
        count_steps += 1

    print(
        f"Episode {episode+1} in scene {scene_name} finished after {count_steps} steps."
    )
    writer.close()

    if episode == 2:
        quit()
