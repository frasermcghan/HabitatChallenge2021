import math

import matplotlib.pyplot as plt
import numpy as np

import habitat
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat_baselines.agents.slam_agents import ORBSLAM2Agent
from habitat_baselines.config.default import get_config
from habitat_baselines.utils.env_utils import make_env_fn


def rotate_origin_only(xy, radians):
    """Only rotate a point around the origin (0, 0)."""
    x, y = xy
    xx = x * math.cos(radians) + y * math.sin(radians)
    yy = -x * math.sin(radians) + y * math.cos(radians)

    return xx, yy


baseline_config = get_config("configs/tasks/pointnav_rgbd.yaml")
config = habitat.get_config("configs/tasks/pointnav_rgbd.yaml")

config.defrost()
baseline_config.defrost()

config.TASK_CONFIG = baseline_config.TASK_CONFIG
config.ORBSLAM2 = baseline_config.ORBSLAM2
config.ORBSLAM2.DIST_TO_STOP = 0.20
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

agent = ORBSLAM2Agent(config.ORBSLAM2)

env_class = habitat.Env
env = make_env_fn(config, env_class)

follower = ShortestPathFollower(
    env.sim, goal_radius=0.05, return_one_hot=False
)

# n_episodes = env.number_of_episodes
n_episodes = 50

total_error = 0.0
success = 0.0
spl = 0.0
for episode in range(n_episodes):

    true_pos = []
    slam_pos = []
    true_headings = []

    count_steps = 0

    observations = env.reset()
    agent.reset()

    scene_name = env.current_episode.scene_id.split(sep="/")[-1].strip(".glb")
    print(f"\nStarting episode {episode+1}/{n_episodes} in {scene_name}...")

    print("Stepping around...")
    while not env.episode_over:

        true_headings.append(observations["heading"])

        pos = env.sim.get_agent_state().position
        true_pos.append([pos[0], pos[2]])

        agent_pose = agent.pose6D.detach().cpu().numpy().reshape(4, 4)
        slam_pos.append([agent_pose[0, -1], agent_pose[2, -1]])

        orb_action = agent.act(observations)
        goal = env.current_episode.goals[0].position
        action = follower.get_next_action(goal_pos=goal)

        observations = env.step(orb_action)
        count_steps += 1

    print(
        f"Episode {episode+1} in scene {scene_name} finished after {count_steps} steps."
    )

    init_heading = true_headings[0]
    init_pos = true_pos[0]

    habitat_to_slam = []

    for position in true_pos:

        x = position[0] - init_pos[0]
        z = -(position[1] - init_pos[1])
        translated = [x, z]
        rotated = rotate_origin_only(translated, radians=init_heading)
        habitat_to_slam.append(rotated)

    slam_xs = [p[0] for p in slam_pos]
    slam_zs = [p[1] for p in slam_pos]

    true_xs = [p[0] for p in habitat_to_slam]
    true_zs = [p[1] for p in habitat_to_slam]

    mean_error = np.mean(np.abs(np.array(true_pos) - np.array(slam_pos)))
    total_error += mean_error
    success += env.get_metrics()["success"]
    spl += env.get_metrics()["spl"]

    # plt.gca().set_aspect("equal", adjustable="datalim")
    plt.gca().set_aspect("equal")
    plt.title(f"{scene_name} {episode}")
    plt.plot(true_xs, true_zs, label="GT", color="red")
    plt.plot(slam_xs, slam_zs, "b--", label="slam")
    plt.legend(loc="best")
    plt.savefig(f"saved/orbslam_actions/trajectory{episode}.png")
    plt.cla()

mean_error = total_error / n_episodes
print(f"Mean SLAM error over {n_episodes} episodes: {mean_error}")
print(f"SPL: {spl/n_episodes}, Success: {success/n_episodes}")
