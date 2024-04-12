
import numpy as np
import carla

from collections import deque
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from custom_leaderboard.autoagents.autonomous_agent import AutonomousAgent

from envlib.reward import RewardCounter, EgoEnvState
from utils.planner import RoutePlanner
from utils.bev_generator import BEV_Generator
from utils.config_parser import AgentConfig


def get_entry_point():
    return "BaseInteractor"

class BaseInteractor(AutonomousAgent):
    def setup(self, path_to_conf_file, EnvQueue, ActionQueue):
        super().setup(path_to_conf_file)

        AgentConfig.parse_config(path_to_conf_file)
        self.EnvQueue, self.ActionQueue = EnvQueue, ActionQueue

        self._vehicle = CarlaDataProvider.get_hero_actor()
        self._world = self._vehicle.get_world()
        self._map = self._world.get_map()
        
        self.reset()

    def reset(self):
        self.control = carla.VehicleControl()
        self.pedal = 0.6
        self.action = np.array([0,0])

        _len = 5
        self.measurement_deque = deque(np.repeat(np.zeros((1,9+2+2+2)), _len, axis=0), maxlen=_len)

        self._planner = RoutePlanner(4, 20, AgentConfig.route_window_size)
        self._planner.set_route(self._global_plan, self._global_plan_world_coord)

    def sensors(self):
        sensor_list = [
            # {  # bev_seg_camera
            #     "type": "sensor.camera.semantic_segmentation",
            #     "x": 0.0,
            #     "y": 0.0,
            #     "z": 8.0,
            #     "roll": 0.0,
            #     "pitch": -90.0,
            #     "yaw": 0.0,
            #     "width": AgentConfig.sensor_seg_width,
            #     "height": AgentConfig.sensor_seg_height,
            #     "fov": 10 * 10.0,
            #     "id": "bev_seg",
            # },
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
        state["route_states"] = self._get_route_states(ego_pos, compass, waypoints)
        state["err_states"] = np.array([EgoEnvState.error_dis, EgoEnvState.error_angle])

        # speed states
        speed = np.round(input_data["speedometer"][1]["speed"], 4) / 20
        state["speed"] = np.ones(AgentConfig.speed_resample) * speed

        # light states
        light_state = self._vehicle.get_traffic_light_state()
        if light_state == carla.TrafficLightState.Red:
            state["light"] = np.ones(AgentConfig.red_light_resample)
        else:
            state["light"] = np.zeros(AgentConfig.red_light_resample)

        # stop sign states
        actors = self._world.get_actors()
        self._affected_by_stop = False
        stop_sign = self._is_stop_sign_hazard(actors.filter("*stop*"))
        if stop_sign:
            state["stop"] = np.ones(AgentConfig.stop_sign_resample)
        else:
            state["stop"] = np.zeros(AgentConfig.stop_sign_resample)

        state["last_action"] = np.repeat(
            np.array([self.control.steer, self.pedal]),
            AgentConfig.last_action_resample)
       
        bev = BEV_Generator(self._world, self._vehicle, compass, waypoints[1, -1])
        # state["bev_feature"] = bev.step()

        measurement = np.concatenate([feature for feature in state.values()])
        self.measurement_deque.append(measurement)
        measurement = np.concatenate(self.measurement_deque)
        
        state = {'measurement': measurement, 'bev': bev.step()}

        return state

    def run_step(self, input_data, timestamp):
        state = self.get_state(input_data)
        reward, terminal = RewardCounter.step()
        action_mask = (
            np.sum(
                np.array([[self.pedal < -0.99],
                          [self.pedal > 0.99],
                          [self.control.steer < -0.99],
                          [self.control.steer > 0.99],
                        ]) * AgentConfig.action_mask_basis,
                axis=0)
        )
        self.EnvQueue.put((state, action_mask, reward, terminal))
        self.action = self.ActionQueue.get()[:2]
        
        steer = self.control.steer + float(self.action[0])
        self.control.steer = max(min(steer, 1), -1)

        self.pedal = min(1,max(-1,self.pedal + float(self.action[1])))

        if self.pedal > 0:
            self.control.throttle = self.pedal
            self.control.brake = 0.0
        else:
            self.control.throttle = 0.0
            self.control.brake = -self.pedal

        return self.control

    def _get_route_states(self, ego_pos, theta, waypoints):
        """
        get route states.
        - waypoints: shape(n,3) -> n*[x, y, lane_command]
        - ego_pos: shape(2, ) -> [x,y]
        - theta: scalar -> angle of the hero vehicle
        """

        scale_factor = 200
        R = np.array(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ]
        )

        aim = (waypoints[:, :2] - ego_pos).dot(R)
        angle = np.arctan2(aim[:, 0], -aim[:, 1])
        distance = np.linalg.norm(aim, axis=1) / scale_factor
        
        geo_feature = np.repeat([angle, distance], AgentConfig.route_resample)

        return np.concatenate([geo_feature, waypoints[:, 2]])
    
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
