#!/usr/bin/env python

import argparse
from argparse import RawTextHelpFormatter
import time
import torch
import numpy as np
import multiprocessing
import wandb

from leaderboard.utils.statistics_manager import StatisticsManager

from custom_evaluator import LeaderboardEvaluator
from models.custom_sac import SACConfig, SAC, DSAC
from utils.metrics import AverageMeter
from utils.config_parser import AgentConfig

class CARLA_Environment(LeaderboardEvaluator):

    def __init__(self, args, statistics_manager, EnvQueue, ActionQueue):
        super().__init__(args, statistics_manager)
        self.EnvQueue = EnvQueue
        self.ActionQueue = ActionQueue

    def _setup_agent(self, args):
        agent_class_name = getattr(self.module_agent, "get_entry_point")()
        agent_class_obj = getattr(self.module_agent, agent_class_name)

        # Start the ROS1 bridge server only for ROS1 based agents.
        if getattr(agent_class_obj, "get_ros_version")() == 1 and self._ros1_server is None:
            from leaderboard.autoagents.ros1_agent import ROS1Server

            self._ros1_server = ROS1Server()
            self._ros1_server.start()

        self.agent_instance = agent_class_obj(args.host, args.port, args.debug)
        self.agent_instance.set_global_plan(
            self.route_scenario.gps_route, self.route_scenario.route
        )
        self.agent_instance.setup(args.agent_config,
                                  self.EnvQueue, self.ActionQueue)

        # Check and store the sensors
        if not self.sensors:
            self.sensors = self.agent_instance.sensors()
            track = self.agent_instance.track

            # validate_sensor_configuration(self.sensors, track, args.track)

            self.sensor_icons = [
                self.sensors_to_icons[sensor["type"]] for sensor in self.sensors
            ]
            self.statistics_manager.save_sensors(self.sensor_icons)
            self.statistics_manager.write_statistics()

            self.sensors_initialized = True
    
def train(arguments):
    AgentConfig.parse_config('./configs/config_multi.yaml')
    
    envs_num = 3
    gpu_index = arguments.gpu_index

    EnvQueue = [multiprocessing.Queue()] * envs_num
    ActionQueue = [multiprocessing.Queue()] * envs_num
    procs = []
    carla_ports = [(arguments.port +20*gpu_index +3*i,
                    arguments.traffic_manager_port +20*gpu_index +3*i) for i in range(envs_num)]

    route_subsets = AgentConfig.exp_config_dict['route_sets']
    for i in range(envs_num):
        proc = multiprocessing.Process(target = load_single_simulation, 
                                       args=(arguments, 
                                             carla_ports[i], route_subsets[i],
                                             EnvQueue[i], ActionQueue[i]))
        proc.start()
        procs.append(proc)

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_wandb = AgentConfig.exp_config_dict['log_wandb']
    if log_wandb:
        wandb.init(
            project = "MultiScenario",
            name = f"exp_{timestamp}",
            config = {**AgentConfig.exp_config_dict, 
                      **AgentConfig.agent_config_dict}
        )
        wandb.save('./envlib/reward.py', policy='now')
        # wandb.watch(agent.actor_net, log="all", log_freq=5000, idx = 0)
    # wandb.watch(agent.critic_net1, log="all", log_freq=50, idx = 1)
    # wandb.watch(agent.critic_net2, log="all", log_freq=50, idx = 2)

    
    agent_config = SACConfig(AgentConfig.agent_config_dict)
    if AgentConfig.agent_name == 'DiscreteAgent':
        agent = DSAC(agent_config, device=torch.device("cuda"))
        train_discrete(agent,
                       envs_num,
                       log_wandb,
                       timestamp,
                       EnvQueue, ActionQueue)
    else:
        agent = SAC(agent_config, device=torch.device("cuda"))
        train_continuous(agent,
                         envs_num,
                         log_wandb,
                         timestamp,
                         EnvQueue, ActionQueue)

def train_discrete(agent, envs_num, log_wandb, timestamp, EnvQueue, ActionQueue):

    actor_losses = AverageMeter()
    q1_losses = AverageMeter()
    q2_losses = AverageMeter()
    scores = [AverageMeter()] *envs_num

    record_interval = AgentConfig.exp_config_dict['record_interval']
    save_interval = AgentConfig.exp_config_dict['save_interval']

    step = 0
    states = []
    action_masks = []
    episodes = np.zeros(envs_num, dtype=np.int32)

    for i in range(envs_num):
        transition = EnvQueue[i].get()
        states.append(transition[0])
        action_masks.append(transition[1])

    agent.set_encoder(states[0])

    run_flag = True
    episode_reset = False
    while run_flag:
        try:
            step += 1
            for i in range(envs_num):
                action, action_encoded = agent.get_action(states[i], action_masks[i])             
                ActionQueue[i].put(action)
                next_state, next_mask, reward_dict, terminal = EnvQueue[i].get()
                # skip reset frame without pushing the transition
                if episode_reset:
                    states[i] = next_state
                    action_masks[i] = next_mask
                    episode_reset = False
                    continue

                reward_dict['control_diff'] = 0
                if abs(action[0]) < 0.01:
                    reward_dict['control_diff'] += 0.1
                if abs(action[1]) < 0.01:
                    reward_dict['control_diff'] += 0.1

                reward = sum(reward_dict.values())
                transition = [states[i], action_encoded, next_state, reward, terminal, action_masks[i], next_mask]

                agent.push(transition)
                loss_tuple = agent.train()

                if loss_tuple:
                    actor_loss, q1_loss, q2_loss = loss_tuple
                    actor_losses.update(actor_loss)
                    q1_losses.update(q1_loss)
                    q2_losses.update(q2_loss)

                states[i] = next_state
                action_masks[i] = next_mask
                scores[i].update(reward)

                if step % record_interval == 0 and log_wandb:
                    wandb.log({
                        "main/actor_loss": actor_losses.avg,
                        "main/critic1_loss": q1_losses.avg,
                        "main/critic2_loss": q2_losses.avg,
                        "main/alpha": agent.alpha.item(),
                        f"control/{i}_pedal_diff": action[1],
                        f"control/{i}_steer_diff": action[0],
                        **{'reward/'+ key: reward_dict[key] for key in reward_dict},
                        'reward/reward_step': sum([reward_dict[key] for key in reward_dict if key != 'Terminal'])
                        }, commit=not terminal)   

                if terminal:
                    episodes[i] += 1
                    if log_wandb:
                        wandb.log({f"reward/episode_reward_{i}": scores[i].sum})
                    if episodes.sum() % save_interval == 0:
                        save_path = f"./logs/weights/{timestamp}_ep{episodes.sum()}"
                        print(f"\033[33m- Saving checkpoint to {save_path} \033[0m")
                        agent.save(save_path)
                    scores[i].reset()
                    episode_reset = True                   
                    continue                  

        except KeyboardInterrupt:
            if log_wandb:
                wandb.finish()
            run_flag = False

def train_continuous(agent, envs_num, log_wandb, timestamp, EnvQueue, ActionQueue):
    actor_losses = AverageMeter()
    q1_losses = AverageMeter()
    q2_losses = AverageMeter()
    scores = [AverageMeter()] *envs_num

    record_interval = AgentConfig.exp_config_dict['record_interval']
    save_interval = AgentConfig.exp_config_dict['save_interval']

    step = 0
    states = []
    episodes = np.zeros(envs_num, dtype=np.int32)

    for i in range(envs_num):
        states.append(EnvQueue[i].get()[0])

    agent.set_encoder(states[0])

    run_flag = True
    episode_reset = False
    while run_flag:
        try:
            step += 1
            for i in range(envs_num):
                action = agent.get_action(states[i])             
                ActionQueue[i].put(action)
                next_state, reward_dict, terminal = EnvQueue[i].get()
                # skip reset frame without pushing the transition
                if episode_reset:
                    states[i] = next_state
                    episode_reset = False
                    continue

                reward = sum(reward_dict.values())
                transition = [states[i], action, next_state, reward, terminal]

                agent.push(transition)
                loss_tuple = agent.train()

                if loss_tuple:
                    actor_loss, q1_loss, q2_loss = loss_tuple
                    actor_losses.update(actor_loss)
                    q1_losses.update(q1_loss)
                    q2_losses.update(q2_loss)

                states[i] = next_state
                scores[i].update(reward)

                if step % record_interval == 0 and log_wandb:
                    wandb.log({
                        "main/actor_loss": actor_losses.avg,
                        "main/critic1_loss": q1_losses.avg,
                        "main/critic2_loss": q2_losses.avg,
                        "main/alpha": agent.alpha.item(),
                        f"control/{i}_pedal": action[1],
                        f"control/{i}_steer": action[0],
                        **{'reward/'+ key: reward_dict[key] for key in reward_dict},
                        'reward/reward_step': sum([reward_dict[key] for key in reward_dict if key != 'Terminal'])
                        }, commit=not terminal)   

                if terminal:
                    episodes[i] += 1
                    if log_wandb:
                        wandb.log({f"reward/episode_reward_{i}": scores[i].sum})
                    if episodes.sum() % save_interval == 0:
                        save_path = f"./logs/weights/{timestamp}_ep{episodes.sum()}"
                        print(f"\033[33m- Saving checkpoint to {save_path} \033[0m")
                        agent.save(save_path)
                    scores[i].reset()
                    episode_reset = True                 

        except KeyboardInterrupt:
            if log_wandb:
                wandb.finish()
            run_flag = False


def load_single_simulation(arguments, ports, routes, EnvQueue, ActionQueue):
    statistics_manager = StatisticsManager(arguments.checkpoint, arguments.debug_checkpoint)
    arguments.port = ports[0]
    arguments.traffic_manager_port = ports[1]
    arguments.routes_subset = routes

    run_flag = True
    while run_flag:
        try:
            env = CARLA_Environment(arguments, statistics_manager, EnvQueue, ActionQueue)
            env.run(arguments)
        except KeyboardInterrupt:
            run_flag = False
            del env

def main():
    description = "CARLA AD Leaderboard Evaluation: evaluate your Agent in CARLA scenarios\n"

    # general parameters
    parser = argparse.ArgumentParser(description=description, formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "--host", default="localhost", help="IP of the host server (default: localhost)"
    )
    parser.add_argument(
        "--port", default=2000, type=int, help="TCP port to listen to (default: 2000)"
    )
    parser.add_argument(
        "--traffic-manager-port",
        default=8000,
        type=int,
        help="Port to use for the TrafficManager (default: 8000)",
    )
    parser.add_argument(
        "--traffic-manager-seed",
        default=0,
        type=int,
        help="Seed used by the TrafficManager (default: 0)",
    )
    parser.add_argument("--debug", type=int, help="Run with debug output", default=0)
    parser.add_argument(
        "--record",
        type=str,
        default="",
        help="Use CARLA recording feature to create a recording of the scenario",
    )
    parser.add_argument(
        "--timeout", default=80.0, type=float, help="Set the CARLA client timeout value in seconds"
    )

    # simulation setup
    parser.add_argument("--routes", required=True, help="Name of the routes file to be executed.")
    parser.add_argument(
        "--routes-subset", default="", type=str, help="Execute a specific set of routes"
    )
    parser.add_argument(
        "--repetitions", type=int, default=1, help="Number of repetitions per route."
    )

    # agent-related options
    parser.add_argument(
        "-a", "--agent", type=str, help="Path to Agent's py file to evaluate", required=True
    )
    parser.add_argument(
        "--agent-config", type=str, help="Path to Agent's configuration file", default=""
    )

    parser.add_argument(
        "--track", type=str, default="SENSORS", help="Participation track: SENSORS, MAP"
    )
    parser.add_argument(
        "--resume", type=bool, default=False, help="Resume execution from last checkpoint?"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./logs/simulation_results.json",
        help="Path to checkpoint used for saving statistics and resuming",
    )
    parser.add_argument(
        "--debug-checkpoint",
        type=str,
        default="./live_results.txt",
        help="Path to checkpoint used for saving live results",
    )
    parser.add_argument(
        "--gpu-index",
        type=int
    )

    arguments = parser.parse_args()

    train(arguments)

if __name__ == "__main__":
    main()
