import time
import numpy as np
import torch
import wandb
import pickle
from collections import deque

import carla
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from custom_leaderboard.autoagents.autonomous_agent import AutonomousAgent

from envlib.reward import RewardCounter
from utils.planner import RoutePlanner
from utils.metrics import AverageMeter
from utils.sensor_wrapper import SegWrapper
from utils.bev_generator import BEV_Generator
from envlib.reward import EgoEnvState
from utils.config_parser import AgentConfig

from pearl.api.action_result import ActionResult
from pearl.pearl_agent import PearlAgent
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace
from pearl.replay_buffers.sequential_decision_making.fifo_off_policy_replay_buffer import FIFOOffPolicyReplayBuffer
from pearl.replay_buffers.sequential_decision_making.fifo_on_policy_replay_buffer import FIFOOnPolicyReplayBuffer
from pearl.policy_learners.sequential_decision_making.soft_actor_critic import SoftActorCritic
from pearl.policy_learners.sequential_decision_making.soft_actor_critic_continuous import ContinuousSoftActorCritic
from pearl.policy_learners.sequential_decision_making.ppo import ProximalPolicyOptimization

# from models.pearl import MultiStateReplayBuffer

def get_entry_point():
    return AgentConfig.agent_name

def get_algorithm():
    algorithm = AgentConfig.exp_config_dict['algorithm']
    return {'SAC': ContinuousSoftActorCritic,
            'DSAC': SoftActorCritic,
            'PPO': ProximalPolicyOptimization}[algorithm]

def get_buffer():
    algorithm = AgentConfig.exp_config_dict['algorithm']
    return {'SAC': FIFOOffPolicyReplayBuffer,
            'DSAC': FIFOOffPolicyReplayBuffer,
            'PPO': FIFOOnPolicyReplayBuffer}[algorithm]

class BaseAgent(AutonomousAgent):
    def setup(self, path_to_conf_file):
        super().setup(path_to_conf_file)

        self._vehicle = CarlaDataProvider.get_hero_actor()
        self._world = self._vehicle.get_world()
        self._map = self._world.get_map()
        config_dict = AgentConfig.exp_config_dict

        self.log_wandb = config_dict['log_wandb']
        self.timestamp = time.strftime("%m%d_%H%M%S", time.localtime())
        scenario_name = AgentConfig.scenario_name
        if scenario_name is None:
            scenario_name = "StraightForward"
        if self.log_wandb:
            wandb.init(
                project = config_dict['project'],
                name = f"{scenario_name}_{self.timestamp}",
                entity = config_dict['entity'],
                config = {**config_dict,
                          **AgentConfig.agent_config_dict,
                          'start_steps': AgentConfig.start_steps,
                          'buffer_size': AgentConfig.buffer_size}
            )
            wandb.save('./envlib/reward.py', policy='now')
            # wandb.watch(self._agent.actor_net, log="all", log_freq=5000, idx = 0)

        self.action_type = config_dict['action_type']
        self.action_mask = config_dict['action_mask']
        self.record_interval = config_dict['record_interval']
        self.save_interval = config_dict['save_interval']

        self.episode = 0
        self.control_step = -1
        self.reset()

    def reset(self):
        self.step = 0
        self.reset_flag = True
        self.route_feature = None

        self.control = carla.VehicleControl()
        self.speed = 0
        self.pedal = 0.6

        self._planner = RoutePlanner(4, 20, AgentConfig.route_window_size)
        self._planner.set_route(self._global_plan, self._global_plan_world_coord)
        
        _len = AgentConfig.state_horizen
        _dim = AgentConfig.measurement_dim // _len
        self.measurement_deque = deque(
            np.repeat(np.zeros((1, _dim)), _len, axis=0), maxlen=_len)

    @classmethod
    def _counter(cls, run_step_func):
        def _run_step(self, *args, **kwargs):
            self.step += 1      
            if self.step % AgentConfig.control_interval == 0:
                return run_step_func(self, *args, **kwargs)
            else:
                return self.control 
        return _run_step

    def sensors(self):
        sensor_list = [
            {
                "type": "sensor.other.imu",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "sensor_tick": 0.05,
                "id": "imu",
            },
            {
                "type": "sensor.other.gnss",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "sensor_tick": 0.01,
                "id": "gps",
            },
            {"type": "sensor.speedometer", "reading_frequency": 20, "id": "speedometer"},
        ]
        return sensor_list

    def get_state(self, input_data):
        """
        process input data derived from func run_step()
        """
        state = {}

        gps = input_data["gps"][1][:2]
        compass = input_data["imu"][1][-1]

        # route states
        waypoints = self._planner.run_step(gps)
        ego_pos = gps[::-1] * self._planner.T_gw
        self._update_route_states(ego_pos, compass, waypoints)
        
        state["route_states"]= self.route_feature
        state["err_states"] = np.array([EgoEnvState.error_dis, EgoEnvState.error_angle])
        # print('> showing error:', state["err_states"])
        # print('> showing route_aim:', EgoEnvState.route_aim)

        # speed feature
        self.speed = np.round(input_data["speedometer"][1]["speed"], 4) / 5
        state["speed"] = np.ones(AgentConfig.speed_resample) * self.speed

        # light feature
        light_state = self._vehicle.get_traffic_light_state()
        if light_state == carla.TrafficLightState.Red:
            state["light"] = np.ones(AgentConfig.red_light_resample)
        else:
            state["light"] = np.zeros(AgentConfig.red_light_resample)

        # stop sign feature
        actors = self._world.get_actors()
        self._affected_by_stop = False
        stop_sign = self._is_stop_sign_hazard(actors.filter("*stop*"))
        if stop_sign:
            state["stop"] = np.ones(AgentConfig.stop_sign_resample)
        else:
            state["stop"] = np.zeros(AgentConfig.stop_sign_resample)

        # last action feature
        state["last_action"] = np.repeat(
            np.array([self.control.steer, self.pedal]),
            AgentConfig.last_action_resample) 
        
        # bev feature
        bev = BEV_Generator(self._world, self._vehicle, compass, waypoints[1, -1])

        measurement = np.concatenate([feature for feature in state.values()])
        self.measurement_deque.append(measurement)
        measurement = np.concatenate(self.measurement_deque)

        # return np.concatenate(measurement)
        state = {'measurement': measurement, 'bev': bev.step()}
        # reshape to fit replay buffer
        state = np.concatenate([feature.reshape(-1) for feature in state.values()])

        return state.astype(np.float32)

    def _update_route_states(self, ego_pos, theta, waypoints):
        """
        get route states.
        - waypoints: shape(n,3) -> n*[x, y, lane_command]
        - ego_pos: shape(2, ) -> [x,y]
        - theta: scalar -> angle of the hero vehicle
        """

        if np.isnan(ego_pos).any() or np.isnan(waypoints).any():
            print('> Warning: NaN detected in GNSS sensor data!')
            print('> showing invalid sensor data:', ego_pos, waypoints)
            return

        R = np.array(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ]
        )

        aim = (waypoints[:, :2] - ego_pos).dot(R)
        angle = np.arctan2(aim[:, 0], -aim[:, 1])
        EgoEnvState.route_aim = angle[0]
        # print('>showing aim error:', angle[0])
        distance = np.linalg.norm(aim, axis=1) / 200

        geo_feature = np.repeat([angle, distance], AgentConfig.route_resample)
        self.route_feature = np.concatenate([geo_feature, waypoints[:, 2]/4])

    
    def _point_inside_boundingbox(self, point, bb_center, bb_extent):
        A = carla.Vector2D(bb_center.x - bb_extent.x, bb_center.y - bb_extent.y)
        B = carla.Vector2D(bb_center.x + bb_extent.x, bb_center.y - bb_extent.y)
        D = carla.Vector2D(bb_center.x - bb_extent.x, bb_center.y + bb_extent.y)
        M = carla.Vector2D(point.x, point.y)

        AB = B - A
        AD = D - A
        AM = M - A
        am_ab = AM.x * AB.x + AM.y * AB.y
        ab_ab = AB.x * AB.x + AB.y * AB.y
        am_ad = AM.x * AD.x + AM.y * AD.y
        ad_ad = AD.x * AD.x + AD.y * AD.y

        return am_ab > 0 and am_ab < ab_ab and am_ad > 0 and am_ad < ad_ad

    def _is_actor_affected_by_stop(self, actor, stop, multi_step=20):
        """
        Check if the given actor is affected by the stop
        """
        affected = False
        PROXIMITY_THRESHOLD = 30.0  # meters
        WAYPOINT_STEP = 1.0  # meters

        # first we run a fast coarse test
        current_location = actor.get_location()
        stop_location = stop.get_transform().location
        if stop_location.distance(current_location) > PROXIMITY_THRESHOLD:
            return affected

        stop_t = stop.get_transform()
        transformed_tv = stop_t.transform(stop.trigger_volume.location)

        # slower and accurate test based on waypoint's horizon and geometric test
        list_locations = [current_location]
        waypoint = self._map.get_waypoint(current_location)
        for _ in range(multi_step):
            if waypoint:
                waypoint = waypoint.next(WAYPOINT_STEP)[0]
                if not waypoint:
                    break
                list_locations.append(waypoint.transform.location)

        for actor_location in list_locations:
            if self._point_inside_boundingbox(
                actor_location, transformed_tv, stop.trigger_volume.extent
            ):
                affected = True

        return affected

    def _is_stop_sign_hazard(self, stop_sign_list):
        res = []
        SPEED_THRESHOLD = 0.1

        if self._affected_by_stop:
            if not self._stop_completed:
                current_speed = self._vehicle.get_velocity()
                if current_speed < SPEED_THRESHOLD:
                    self._stop_completed = True
                    return res
                else:
                    return [self._target_stop_sign]
            else:
                # reset if the ego vehicle is outside the influence of the current stop sign
                if not self._is_actor_affected_by_stop(self._vehicle, self._target_stop_sign):
                    self._affected_by_stop = False
                    self._stop_completed = False
                    self._target_stop_sign = None
                return res

        ve_tra = self._vehicle.get_transform()
        ve_dir = ve_tra.get_forward_vector()

        wp = self._map.get_waypoint(ve_tra.location)
        wp_dir = wp.transform.get_forward_vector()

        dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

        if dot_ve_wp > 0:  # Ignore all when going in a wrong lane
            for stop_sign in stop_sign_list:
                if self._is_actor_affected_by_stop(self._vehicle, stop_sign):
                    # this stop sign is affecting the vehicle
                    self._affected_by_stop = True
                    self._target_stop_sign = stop_sign
                    res.append(self._target_stop_sign)

        return res

class MaskAgent(BaseAgent):
    def setup(self, path_to_conf_file):
        super().setup(path_to_conf_file)

        self._agent = PearlAgent(
            policy_learner=get_algorithm()(
                **AgentConfig.agent_config_dict
            ),
            replay_buffer=get_buffer()(AgentConfig.buffer_size)
        )
        # if self.log_wandb:
        #     wandb.watch(self._agent.policy_learner._actor, log="all", log_freq=100, idx = 0)
        #     wandb.watch(self._agent.policy_learner._critic, log="all", log_freq=100, idx = 1)
        self.scores = AverageMeter()

        load_path = AgentConfig.exp_config_dict['ckpt_name']
        self.exploit = AgentConfig.exp_config_dict['eval']
        if load_path is not None:
            with open('./logs/weights/' + load_path, 'rb') as file:
                self._agent = pickle.load(file)
            

    @BaseAgent._counter
    def run_step(self, input_data, timestamp):
        self.control_step += 1
        # self.control.throttle = 0.5
        # # print('> road completion:', Ego)
        # return self.control
        if self.reset_flag:
            self.state = self.get_state(input_data)
            action_space = self._get_valid_action()
            self._agent.reset(self.state, action_space)
            self.reset_flag = False
        else:
            reward_dict, terminal = RewardCounter.step()
            reward = sum(reward_dict.values())
            action_space = self._get_valid_action()
            self._agent.observe(ActionResult(observation=self.state,
                                             reward=float(reward),
                                             terminated=terminal,
                                             truncated=terminal,
                                             available_action_space=action_space))

            report = {}
            if self.control_step > AgentConfig.start_steps:
                report = self._agent.learn()
            self.scores.update(reward)
            # print('>showing aim:', EgoEnvState.route_aim)

            if self.control_step % self.record_interval == 0 and self.log_wandb:
                wandb.log({
                    **{'main/'+ key: sum(value)/len(value) for key, value in report.items()},
                    "control/pedal": self.pedal,
                    "control/steer": self.control.steer,
                    "route_state/speed": EgoEnvState.speed,
                    "route_state/aim": EgoEnvState.route_aim,
                    'route_state/completion': EgoEnvState.route_completion,
                    'route_state/finish_time': EgoEnvState.finish_time,
                    **{'reward/'+ key: reward_dict[key] for key in reward_dict},
                    'reward/reward_step': sum([reward_dict[key] for key in reward_dict if key != 'Terminal'])
                    }, commit=not terminal)

            if terminal:
                self.episode += 1
                if self.log_wandb:
                    wandb.log({f"reward/episode_reward": self.scores.sum})
                if self.episode % self.save_interval == 0:
                    save_path = f"./logs/weights/{self.timestamp}_ep{self.episode}.pkl"
                    print(f"\033[33m- Saving agent checkpoint to {save_path} \033[0m")
                    with open(save_path, 'wb') as file:
                        pickle.dump(self._agent, file)
                self.scores.reset()

        action_idx = self._agent.act(self.exploit)
        action = action_space[action_idx]
        self.state = self.get_state(input_data)

        if self.action_type['steer'] == 'delta':
            steer = self.control.steer + float(action[0])
            self.control.steer = max(min(steer, 1), -1)
        else:
            self.control.steer = float(action[0])

        _alpha = 1
        if self.action_type['throttle'] == 'delta':
            pedal = self.pedal + float(action[0])
            self.pedal = max(min(pedal, 1), -1)
        else:
            self.pedal = (1-_alpha) * self.pedal + _alpha * float(action[1])

        if self.pedal > 0:
            self.control.throttle = self.pedal
            self.control.brake = 0.0
        else:
            self.control.throttle = 0.0
            self.control.brake = -self.pedal

        return self.control

    def _get_valid_action(self):
        if not self.action_mask:
            return AgentConfig.agent_config_dict['action_space']
        
        tb = np.array(AgentConfig.throttle_set)
        sb = np.array(AgentConfig.steer_set)

        # mask_type = AgentConfig.action_mask_type
        mask_type = 1

        # available action bool set1 & condition1 |
        # available action bool set2 & condition2 ...
        if self.action_type['throttle'] == 'direct':
            tb = tb[(tb < -0.3) & (self.speed > 10) |
                    (tb > -0.3) & (tb <= 0) & (self.speed > 1) |
                    (tb >= 0) & (self.speed < AgentConfig.speed_limit)]
            if (abs(tb - self.pedal) < 0.31).any():
                tb = tb[(abs(tb - self.pedal) < 0.31)]

        if mask_type == 0:
            aim = EgoEnvState.error_angle
            thres = [0.3, 0.6]
        elif mask_type == 1:
            aim = - EgoEnvState.route_aim
            thres = [0.1, 0.2]
        elif mask_type == 2:
            aim = EgoEnvState.error_angle
            thres = [0.1, 0.2]       
            
        if self.action_type['steer'] =='direct':
            sb = sb[(sb < -0.6) & (aim > -thres[0]) |
                    (sb >= -0.6) & (sb <= 0) & (aim > -thres[1]) |
                    (sb <= 0.6) & (sb >= 0) & (aim < thres[0]) |
                    (sb > 0.6) & (aim < thres[1])]
                
            if (abs(sb - self.control.steer) < 0.31).any():
                sb = sb[(abs(sb - self.control.steer) < 0.31)]
        
        valid_space = torch.from_numpy(
            np.array(
                list(map(
                    np.ravel,
                    np.meshgrid(sb,
                                tb)))
                ).T
        ).float()
        return DiscreteActionSpace(valid_space)
