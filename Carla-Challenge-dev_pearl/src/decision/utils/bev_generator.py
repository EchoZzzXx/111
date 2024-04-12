import carla
import copy
import cv2
import math
import shapely
import numpy as np

from .heatmap_utils import generate_heatmap
from .det_utils import generate_det_data

from agents.navigation.local_planner import RoadOption
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


def _numpy(carla_vector, normalize=False):
    result = np.float32([carla_vector.x, carla_vector.y])
    if normalize:
        return result / (np.linalg.norm(result) + 1e-4)
    return result


def _location(x, y, z):
    return carla.Location(x=float(x), y=float(y), z=float(z))


def _orientation(yaw):
    return np.float32([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])


def get_collision(p1, v1, p2, v2):
    A = np.stack([v1, -v2], 1)
    b = p2 - p1

    if abs(np.linalg.det(A)) < 1e-3:
        return False, None

    x = np.linalg.solve(A, b)
    collides = all(x >= 0) and all(x <= 1)  # how many seconds until collision

    return collides, p1 + x[0] * v1


class BEV_Generator:
    def __init__(self, world, vehicle, compass, command):
        self._world = world
        self._map = self._world.get_map()
        self._vehicle = vehicle
        self._command = command
        self._compass = compass

        lights_list = self._world.get_actors().filter("*traffic_light*")
        self._map = self._world.get_map()
        self._list_traffic_lights = []
        for light in lights_list:
            center, waypoints = self.get_traffic_light_waypoints(light)
            self._list_traffic_lights.append((light, center, waypoints))
        (
            self._list_traffic_waypoints,
            self._dict_traffic_lights,
        ) = self._gen_traffic_light_dict(self._list_traffic_lights)

        self._affected_by_stop = False

    def step(self):
        self._observe_actors()

        heatmap = generate_heatmap(
            copy.deepcopy(self.measurements), copy.deepcopy(self.actors_data), pixels_per_meter=10
        )

        # det_data = (
        #     generate_det_data(
        #         heatmap,
        #         copy.deepcopy(self.measurements),
        #         copy.deepcopy(self.actors_data),
        #         pixels_per_meter=10,
        #     )
        #     .reshape(400, -1)
        #     .astype(np.float32)
        #     .reshape(-1)
        # )
        

        det_data = generate_det_data(
                heatmap, 
                copy.deepcopy(self.measurements),
                copy.deepcopy(self.actors_data),
                pixels_per_meter=10
            ).astype(np.float32)

        return det_data

        # prob_img = np.uint8(det_data[..., 0] * 255)
        # img_traffic = heatmap[:200, 80:280, None]

        # cv2.imshow('heatmap', heatmap)
        # cv2.imshow('det_data', prob_img)
        # cv2.waitKey(2)
        # img_traffic = transforms.ToTensor()(img_traffic)

    def _observe_actors(self):
        command = self._command
        actors = self._world.get_actors()

        vehicle = self._is_vehicle_hazard(actors.filter("*vehicle*"), command)
        lane_vehicle = self._is_lane_vehicle_hazard(actors.filter("*vehicle*"), command)
        junction_vehicle = self._is_junction_vehicle_hazard(actors.filter("*vehicle*"), command)
        # light = self._is_light_red(actors.filter("*traffic_light*"))
        walker = self._is_walker_hazard(actors.filter("*walker*"))
        bike = self._is_bike_hazard(actors.filter("*vehicle*"))

        # record the reason for braking
        self.is_vehicle_present = [x.id for x in vehicle]
        self.is_lane_vehicle_present = [x.id for x in lane_vehicle]
        self.is_junction_vehicle_present = [x.id for x in junction_vehicle]
        self.is_pedestrian_present = [x.id for x in walker]
        self.is_bike_present = [x.id for x in bike]
        # self.is_red_light_present = [x.id for x in light]

        ego_loc = self._vehicle.get_location()
        self.measurements = {
            "x": ego_loc.x,
            "y": ego_loc.y,
            "theta": self._compass,
            # "is_junction": self.is_junction,
            "is_vehicle_present": self.is_vehicle_present,
            "is_bike_present": self.is_bike_present,
            "is_lane_vehicle_present": self.is_lane_vehicle_present,
            "is_junction_vehicle_present": self.is_junction_vehicle_present,
            "is_pedestrian_present": self.is_pedestrian_present,
            # "is_red_light_present": self.is_red_light_present,
            # "is_stop_sign_present": self.is_stop_sign_present,
        }

        self.actors_data = self._collect_actors_data()

    def _collect_actors_data(self):
        data = {}

        vehicles = self._world.get_actors().filter("*vehicle*")
        for actor in vehicles:
            loc = actor.get_location()
            if loc.distance(self._vehicle.get_location()) > 50:
                continue
            _id = actor.id
            data[_id] = {}
            data[_id]["loc"] = [loc.x, loc.y, loc.z]
            ori = actor.get_transform().rotation.get_forward_vector()
            data[_id]["ori"] = [ori.x, ori.y, ori.z]
            box = actor.bounding_box.extent
            data[_id]["box"] = [box.x, box.y]
            vel = actor.get_velocity()
            data[_id]["vel"] = [vel.x, vel.y, vel.z]
            data[_id]["tpe"] = 0

        walkers = self._world.get_actors().filter("*walker*")
        for actor in walkers:
            loc = actor.get_location()
            if loc.distance(self._vehicle.get_location()) > 50:
                continue
            _id = actor.id
            data[_id] = {}
            data[_id]["loc"] = [loc.x, loc.y, loc.z]
            ori = actor.get_transform().rotation.get_forward_vector()
            data[_id]["ori"] = [ori.x, ori.y, ori.z]
            box = actor.bounding_box.extent
            data[_id]["box"] = [box.x, box.y]
            vel = actor.get_velocity()
            data[_id]["vel"] = [vel.x, vel.y, vel.z]
            data[_id]["tpe"] = 1

        return data
    
    def _gen_traffic_light_dict(self, traffic_lights_list):
        traffic_light_dict = {}
        waypoints_list = []
        for light, center, waypoints in traffic_lights_list:
            for waypoint in waypoints:
                traffic_light_dict[waypoint] = (light, center)
                waypoints_list.append(waypoint)
        return waypoints_list, traffic_light_dict
    
    def get_traffic_light_waypoints(self, traffic_light):
        base_transform = traffic_light.get_transform()
        base_rot = base_transform.rotation.yaw
        area_loc = base_transform.transform(traffic_light.trigger_volume.location)

        # Discretize the trigger box into points
        area_ext = traffic_light.trigger_volume.extent
        x_values = np.arange(
            -0.9 * area_ext.x, 0.9 * area_ext.x, 1.0
        )  # 0.9 to avoid crossing to adjacent lanes

        area = []
        for x in x_values:
            point = self.rotate_point(carla.Vector3D(x, 0, area_ext.z), base_rot)
            point_location = area_loc + carla.Location(x=point.x, y=point.y)
            area.append(point_location)

        # Get the waypoints of these points, removing duplicates
        ini_wps = []
        for pt in area:
            wpx = self._map.get_waypoint(pt)
            # As x_values are arranged in order, only the last one has to be checked
            if (
                not ini_wps
                or ini_wps[-1].road_id != wpx.road_id
                or ini_wps[-1].lane_id != wpx.lane_id
            ):
                ini_wps.append(wpx)

        # Advance them until the intersection
        wps = []
        for wpx in ini_wps:
            while not wpx.is_intersection:
                next_wp = wpx.next(0.5)[0]
                if next_wp and not next_wp.is_intersection:
                    wpx = next_wp
                else:
                    break
            wps.append(wpx)

        return area_loc, wps


    def _find_closest_valid_traffic_light(self, loc, min_dis):
        wp = self._map.get_waypoint(loc)
        min_wp = None
        min_distance = min_dis
        for waypoint in self._list_traffic_waypoints:
            if waypoint.road_id != wp.road_id or waypoint.lane_id * wp.lane_id < 0:
                continue
            dis = loc.distance(waypoint.transform.location)
            if dis <= min_distance:
                min_distance = dis
                min_wp = waypoint
        if min_wp is None:
            return None
        else:
            return self._dict_traffic_lights[min_wp][0]

    def _is_light_red(self, lights_list):
        if self._vehicle.get_traffic_light_state() != carla.libcarla.TrafficLightState.Green:
            affecting = self._vehicle.get_traffic_light()

            for light in self._traffic_lights:
                if light.id == affecting.id:
                    return [light]

        light = self._find_closest_valid_traffic_light(self._vehicle.get_location(), min_dis=8)
        if light is not None and light.state != carla.libcarla.TrafficLightState.Green:
            return [light]
        return []

    def _is_bike_hazard(self, bikes_list):
        res = []
        o1 = _orientation(self._vehicle.get_transform().rotation.yaw)
        v1_hat = o1
        p1 = _numpy(self._vehicle.get_location())
        v1 = 10.0 * o1

        for bike in bikes_list:
            o2 = _orientation(bike.get_transform().rotation.yaw)
            s2 = np.linalg.norm(_numpy(bike.get_velocity()))
            v2_hat = o2
            p2 = _numpy(bike.get_location())

            p2_p1 = p2 - p1
            distance = np.linalg.norm(p2_p1)
            p2_p1_hat = p2_p1 / (distance + 1e-4)

            angle_to_car = np.degrees(np.arccos(np.clip(v1_hat.dot(p2_p1_hat), -1, 1)))
            angle_between_heading = np.degrees(np.arccos(np.clip(o1.dot(o2), -1, 1)))

            # to consider -ve angles too
            angle_to_car = min(angle_to_car, 360.0 - angle_to_car)
            angle_between_heading = min(angle_between_heading, 360.0 - angle_between_heading)
            if distance > 20:
                continue
            if angle_to_car > 30:
                continue
            if angle_between_heading < 80 and angle_between_heading > 100:
                continue

            p2_hat = -2.0 * v2_hat + _numpy(bike.get_location())
            v2 = 7.0 * v2_hat

            collides, collision_point = get_collision(p1, v1, p2_hat, v2)

            if collides:
                res.append(bike)

        return res

    def _is_walker_hazard(self, walkers_list):
        res = []
        p1 = _numpy(self._vehicle.get_location())
        v1 = 10.0 * _orientation(self._vehicle.get_transform().rotation.yaw)

        for walker in walkers_list:
            v2_hat = _orientation(walker.get_transform().rotation.yaw)
            s2 = np.linalg.norm(_numpy(walker.get_velocity()))

            if s2 < 0.05:
                v2_hat *= s2

            p2 = -3.0 * v2_hat + _numpy(walker.get_location())
            v2 = 8.0 * v2_hat

            collides, collision_point = get_collision(p1, v1, p2, v2)

            if collides:
                res.append(walker)

        return res
    
    def _are_vehicles_crossing_future(self, p1, s1, lft_lane, rgt_lane):
        p1_hat = carla.Location(x=p1.x + 3 * s1.x, y=p1.y + 3 * s1.y)
        line1 = shapely.geometry.LineString([(p1.x, p1.y), (p1_hat.x, p1_hat.y)])
        line2 = shapely.geometry.LineString(
            [(lft_lane.x, lft_lane.y), (rgt_lane.x, rgt_lane.y)]
        )
        inter = line1.intersection(line2)
        return not inter.is_empty

    def _is_lane_vehicle_hazard(self, vehicle_list, command):
        res = []
        if command != int(RoadOption.CHANGELANELEFT) and command != int(RoadOption.CHANGELANERIGHT):
            return []

        z = self._vehicle.get_location().z
        w1 = self._map.get_waypoint(self._vehicle.get_location())
        o1 = _orientation(self._vehicle.get_transform().rotation.yaw)
        p1 = self._vehicle.get_location()

        yaw_w1 = w1.transform.rotation.yaw
        lane_width = w1.lane_width
        location_w1 = w1.transform.location

        lft_shift = 0.5
        rgt_shift = 0.5
        if command == int(RoadOption.CHANGELANELEFT):
            rgt_shift += 1
        else:
            lft_shift += 1

        lft_lane_wp = self.rotate_point(
            carla.Vector3D(lft_shift * lane_width, 0.0, location_w1.z), yaw_w1 + 90
        )
        lft_lane_wp = location_w1 + carla.Location(lft_lane_wp)
        rgt_lane_wp = self.rotate_point(
            carla.Vector3D(rgt_shift * lane_width, 0.0, location_w1.z), yaw_w1 - 90
        )
        rgt_lane_wp = location_w1 + carla.Location(rgt_lane_wp)

        for target_vehicle in vehicle_list:
            if target_vehicle.id == self._vehicle.id:
                continue

            w2 = self._map.get_waypoint(target_vehicle.get_location())
            o2 = _orientation(target_vehicle.get_transform().rotation.yaw)
            p2 = target_vehicle.get_location()
            x2 = target_vehicle.bounding_box.extent.x
            p2_hat = p2 - target_vehicle.get_transform().get_forward_vector() * x2 * 2
            s2 = (
                target_vehicle.get_velocity()
                + target_vehicle.get_transform().get_forward_vector() * x2
            )
            s2_value = max(
                12,
                2 + 2 * x2 + 3.0 * np.linalg.norm(_numpy(target_vehicle.get_velocity())),
            )

            distance = p1.distance(p2)

            if distance > s2_value:
                continue
            if w1.road_id != w2.road_id or w1.lane_id * w2.lane_id < 0:
                continue
            if command == int(RoadOption.CHANGELANELEFT):
                if w1.lane_id > 0:
                    if w2.lane_id != w1.lane_id - 1:
                        continue
                if w1.lane_id < 0:
                    if w2.lane_id != w1.lane_id + 1:
                        continue
            if command == int(RoadOption.CHANGELANERIGHT):
                if w1.lane_id > 0:
                    if w2.lane_id != w1.lane_id + 1:
                        continue
                if w1.lane_id < 0:
                    if w2.lane_id != w1.lane_id - 1:
                        continue

            if self._are_vehicles_crossing_future(p2_hat, s2, lft_lane_wp, rgt_lane_wp):
                res.append(target_vehicle)
        return res

    def _is_junction_vehicle_hazard(self, vehicle_list, command):
        res = []
        o1 = _orientation(self._vehicle.get_transform().rotation.yaw)
        x1 = self._vehicle.bounding_box.extent.x
        p1 = self._vehicle.get_location() + x1 * self._vehicle.get_transform().get_forward_vector()
        w1 = self._map.get_waypoint(p1)
        s1 = np.linalg.norm(_numpy(self._vehicle.get_velocity()))
        if command == int(RoadOption.RIGHT):
            shift_angle = 25
        elif command == int(RoadOption.LEFT):
            shift_angle = -25
        else:
            shift_angle = 0
        v1 = (4 * s1 + 5) * _orientation(self._vehicle.get_transform().rotation.yaw + shift_angle)

        for target_vehicle in vehicle_list:
            if target_vehicle.id == self._vehicle.id:
                continue

            o2 = _orientation(target_vehicle.get_transform().rotation.yaw)
            o2_left = _orientation(target_vehicle.get_transform().rotation.yaw - 15)
            o2_right = _orientation(target_vehicle.get_transform().rotation.yaw + 15)
            x2 = target_vehicle.bounding_box.extent.x

            p2 = target_vehicle.get_location()
            p2_hat = p2 - (x2 + 2) * target_vehicle.get_transform().get_forward_vector()
            w2 = self._map.get_waypoint(p2)
            s2 = np.linalg.norm(_numpy(target_vehicle.get_velocity()))

            v2 = (4 * s2 + 2 * x2 + 6) * o2
            v2_left = (4 * s2 + 2 * x2 + 6) * o2_left
            v2_right = (4 * s2 + 2 * x2 + 6) * o2_right

            angle_between_heading = np.degrees(np.arccos(np.clip(o1.dot(o2), -1, 1)))

            if self._vehicle.get_location().distance(p2) > 20:
                continue
            if w1.is_junction == False and w2.is_junction == False:
                continue
            if angle_between_heading < 15.0 or angle_between_heading > 165:
                continue
            collides, collision_point = get_collision(_numpy(p1), v1, _numpy(p2_hat), v2)
            if collides is None:
                collides, collision_point = get_collision(_numpy(p1), v1, _numpy(p2_hat), v2_left)
            if collides is None:
                collides, collision_point = get_collision(_numpy(p1), v1, _numpy(p2_hat), v2_right)

            light = self._find_closest_valid_traffic_light(
                target_vehicle.get_location(), min_dis=10
            )
            if light is not None and light.state != carla.libcarla.TrafficLightState.Green:
                continue
            if collides:
                res.append(target_vehicle)
        return res

    def _is_vehicle_hazard(self, vehicle_list, command):
        res = []
        z = self._vehicle.get_location().z

        o1 = _orientation(self._vehicle.get_transform().rotation.yaw)
        p1 = _numpy(self._vehicle.get_location())
        s1 = max(
            10, 3.0 * np.linalg.norm(_numpy(self._vehicle.get_velocity()))
        )  # increases the threshold distance
        s1a = np.linalg.norm(_numpy(self._vehicle.get_velocity()))
        w1 = self._map.get_waypoint(self._vehicle.get_location())
        v1_hat = o1
        v1 = s1 * v1_hat

        for target_vehicle in vehicle_list:
            if target_vehicle.id == self._vehicle.id:
                continue
            if not target_vehicle.is_alive:
                continue

            o2 = _orientation(target_vehicle.get_transform().rotation.yaw)
            p2 = _numpy(target_vehicle.get_location())
            s2 = max(5.0, 2.0 * np.linalg.norm(_numpy(target_vehicle.get_velocity())))
            s2a = np.linalg.norm(_numpy(target_vehicle.get_velocity()))
            w2 = self._map.get_waypoint(target_vehicle.get_location())
            v2_hat = o2
            v2 = s2 * v2_hat

            p2_p1 = p2 - p1
            distance = np.linalg.norm(p2_p1)
            p2_p1_hat = p2_p1 / (distance + 1e-4)

            angle_to_car = np.degrees(np.arccos(np.clip(v1_hat.dot(p2_p1_hat), -1, 1)))
            angle_between_heading = np.degrees(np.arccos(np.clip(o1.dot(o2), -1, 1)))

            # to consider -ve angles too
            angle_to_car = min(angle_to_car, 360.0 - angle_to_car)
            angle_between_heading = min(angle_between_heading, 360.0 - angle_between_heading)

            if not w2.is_junction and angle_between_heading > 45.0 and s2a < 0.5 and distance > 4:
                if w1.road_id != w2.road_id:
                    continue
            if (
                angle_between_heading < 15
                and w1.road_id == w2.road_id
                and w1.lane_id != w2.lane_id
                and command != int(RoadOption.CHANGELANELEFT)
                and command != int(RoadOption.CHANGELANERIGHT)
            ):
                continue

            if angle_between_heading > 60.0 and not (angle_to_car < 15 and distance < s1):
                continue
            elif angle_to_car > 30.0:
                continue
            elif distance > s1:
                continue

            res.append(target_vehicle)

        return res
    
    def rotate_point(self, point, angle):
        """
        rotate a given point by a given angle
        """
        x_ = (
            math.cos(math.radians(angle)) * point.x
            - math.sin(math.radians(angle)) * point.y
        )
        y_ = (
            math.sin(math.radians(angle)) * point.x
            + math.cos(math.radians(angle)) * point.y
        )
        return carla.Vector3D(x_, y_, point.z)
