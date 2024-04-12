from __future__ import print_function

import carla
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from leaderboard.autoagents import agent_wrapper
from leaderboard.autoagents.ros_base_agent import ROSBaseAgent
from leaderboard.autoagents.agent_wrapper import ROSAgentWrapper

from custom_leaderboard.envs.sensor_interface import CallBack


class AgentWrapper(agent_wrapper.AgentWrapper):
    def _preprocess_sensor_spec(self, sensor_spec):
        type_ = sensor_spec["type"]
        id_ = sensor_spec["id"]
        attributes = {}

        if type_ == "sensor.opendrive_map":
            attributes["reading_frequency"] = sensor_spec["reading_frequency"]
            sensor_location = carla.Location()
            sensor_rotation = carla.Rotation()

        elif type_ == "sensor.speedometer":
            delta_time = CarlaDataProvider.get_world().get_settings().fixed_delta_seconds
            attributes["reading_frequency"] = 1 / delta_time
            sensor_location = carla.Location()
            sensor_rotation = carla.Rotation()

        if type_ == "sensor.camera.rgb" or type_ == "sensor.camera.semantic_segmentation":
            attributes["image_size_x"] = str(sensor_spec["width"])
            attributes["image_size_y"] = str(sensor_spec["height"])
            attributes["fov"] = str(sensor_spec["fov"])

            sensor_location = carla.Location(
                x=sensor_spec["x"], y=sensor_spec["y"], z=sensor_spec["z"]
            )
            sensor_rotation = carla.Rotation(
                pitch=sensor_spec["pitch"], roll=sensor_spec["roll"], yaw=sensor_spec["yaw"]
            )

        elif type_ == "sensor.lidar.ray_cast":
            attributes["range"] = str(85)
            attributes["rotation_frequency"] = str(10)
            attributes["channels"] = str(64)
            attributes["upper_fov"] = str(10)
            attributes["lower_fov"] = str(-30)
            attributes["points_per_second"] = str(600000)
            attributes["atmosphere_attenuation_rate"] = str(0.004)
            attributes["dropoff_general_rate"] = str(0.45)
            attributes["dropoff_intensity_limit"] = str(0.8)
            attributes["dropoff_zero_intensity"] = str(0.4)

            sensor_location = carla.Location(
                x=sensor_spec["x"], y=sensor_spec["y"], z=sensor_spec["z"]
            )
            sensor_rotation = carla.Rotation(
                pitch=sensor_spec["pitch"], roll=sensor_spec["roll"], yaw=sensor_spec["yaw"]
            )

        elif type_ == "sensor.other.radar":
            attributes["horizontal_fov"] = str(sensor_spec["horizontal_fov"])  # degrees
            attributes["vertical_fov"] = str(sensor_spec["vertical_fov"])  # degrees
            attributes["points_per_second"] = "1500"
            attributes["range"] = "100"  # meters

            sensor_location = carla.Location(
                x=sensor_spec["x"], y=sensor_spec["y"], z=sensor_spec["z"]
            )
            sensor_rotation = carla.Rotation(
                pitch=sensor_spec["pitch"], roll=sensor_spec["roll"], yaw=sensor_spec["yaw"]
            )

        elif type_ == "sensor.other.gnss":
            attributes["noise_alt_stddev"] = str(0.000005)
            attributes["noise_lat_stddev"] = str(0.000005)
            attributes["noise_lon_stddev"] = str(0.000005)
            attributes["noise_alt_bias"] = str(0.0)
            attributes["noise_lat_bias"] = str(0.0)
            attributes["noise_lon_bias"] = str(0.0)

            sensor_location = carla.Location(
                x=sensor_spec["x"], y=sensor_spec["y"], z=sensor_spec["z"]
            )
            sensor_rotation = carla.Rotation()

        elif type_ == "sensor.other.imu":
            attributes["noise_accel_stddev_x"] = str(0.001)
            attributes["noise_accel_stddev_y"] = str(0.001)
            attributes["noise_accel_stddev_z"] = str(0.015)
            attributes["noise_gyro_stddev_x"] = str(0.001)
            attributes["noise_gyro_stddev_y"] = str(0.001)
            attributes["noise_gyro_stddev_z"] = str(0.001)

            sensor_location = carla.Location(
                x=sensor_spec["x"], y=sensor_spec["y"], z=sensor_spec["z"]
            )
            sensor_rotation = carla.Rotation(
                pitch=sensor_spec["pitch"], roll=sensor_spec["roll"], yaw=sensor_spec["yaw"]
            )

        sensor_transform = carla.Transform(sensor_location, sensor_rotation)

        return type_, id_, sensor_transform, attributes


class AgentWrapperFactory(object):
    @staticmethod
    def get_wrapper(agent):
        if isinstance(agent, ROSBaseAgent):
            return ROSAgentWrapper(agent)
        else:
            return AgentWrapper(agent)
