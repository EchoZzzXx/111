# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import time
import os
import carla
import argparse
import random
import json
import threading
import pathlib
import glob

import numpy as np

from queue import Queue, Empty
from decision.utils.bev_generator import BEV_Generator
from decision.envlib.reward import step_reward_func, EgoEnvState

## configs
SENSORS = [
    {
        "type": "sensor.other.imu",
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
        "sensor_tick": 0.025,
        "id": "imu",
    }
]

WEATHER = carla.WeatherParameters(
    sun_azimuth_angle=-1.0, sun_altitude_angle=70.0,
    cloudiness=30.0, precipitation=0.0, precipitation_deposits=80.0, wetness=15.0,
    wind_intensity=10.0,
    fog_density=2.0, fog_distance=0.0, fog_falloff=0.0)

# 3) Choose a recorder file (folder path, start time, duration)
# RECORDER_INFO = []
# for path in glob.glob('./data/Scenario_Logs/*')[-1]:
#     RECORDER_INFO.append({'folder': path,
#                           'start_time': 0,
#                           'duration': 0})
RECORDER_INFO = [
    {
        'folder': "./data/Scenario_Logs/NonSignalizedJunctionRightTurn_slow",
        'start_time': 0,
        'duration': 0
    },

    {
        'folder': "./data/Scenario_Logs/SignalizedJunctionLeftTurn_fast",
        'start_time': 0,
        'duration': 0
    },

    {
        'folder': "./data/Scenario_Logs/MergerIntoSlowTraffic",
        'start_time': 0,
        'duration': 0
    },
]

#     {
#         'folder': "./data/Scenario_Logs/MergerIntoSlowTraffic",
#         'start_time': 0,
#         'duration': 0
#     },

#     {
#         'folder': "./data/Scenario_Logs/ParkingCrossingPedestrian",
#         'start_time': 0,
#         'duration': 0
#     },
# ]

# 4) Choose the destination folder where the sensor data will be saved
DESTINATION_FOLDER = "./data/dataset"
################# End user simulation configuration ##################

FPS = 20
THREADS = 10
CURRENT_THREADS = 0

class AgentConfig:
    sensor_seg_width = 50
    sensor_seg_height = 50
    route_window_size = 3
    route_resample = 5
    last_action_resample = 5
    red_light_resample = 0
    stop_sign_resample = 0
    speed_resample = 10

    state_dim = 352


def create_folders(endpoint, sensors):
    return

    sensor_endpoint = f"{endpoint}/measurements"
    if not os.path.exists(sensor_endpoint):
        os.makedirs(sensor_endpoint)
    sensor_endpoint = f"{endpoint}/lidar"
    if not os.path.exists(sensor_endpoint):
        os.makedirs(sensor_endpoint)
    for pos in ["front", "left", "right"]:
        name = "rgb" + "_" + pos
        sensor_endpoint = f"{endpoint}/{name}"
        if not os.path.exists(sensor_endpoint):
            os.makedirs(sensor_endpoint)

    # for sensor_id, sensor_bp in sensors:
    #     sensor_endpoint = f"{endpoint}/{sensor_id}"
    #     if not os.path.exists(sensor_endpoint):
    #         os.makedirs(sensor_endpoint)

        # if 'gps' in sensor_bp:
        #     sensor_endpoint = f"{endpoint}/{sensor_id}/gnss_data.csv"
        #     with open(sensor_endpoint, 'w') as data_file:
        #         data_txt = f"Frame,Altitude,Latitude,Longitude\n"
        #         data_file.write(data_txt)

        # if 'imu' in sensor_bp:
        #     sensor_endpoint = f"{endpoint}/{sensor_id}/imu_data.csv"
        #     with open(sensor_endpoint, 'w') as data_file:
        #         data_txt = f"Frame,Accelerometer X,Accelerometer y,Accelerometer Z,Compass,Gyroscope X,Gyroscope Y,Gyroscope Z\n"
        #         data_file.write(data_txt)

    # create folder for control data
    # control_endpoint = f"{endpoint}/Control"
    # if not os.path.exists(control_endpoint):
    #     os.makedirs(control_endpoint)
    # control_endpoint = f"{endpoint}/Control/control_data.csv"
    # with open(control_endpoint, 'w') as data_file:
    #     data_txt = f"Frame,throttle,steer,brake,hand_brake,reverse,manual_gear_shift,gear\n"
    #     data_file.write(data_txt)def add_listener(sensor, sensor_queue, sensor_id):
            
def add_listener(sensor, sensor_queue, sensor_id):
    sensor.listen(lambda data: sensor_listen(data, sensor_queue, sensor_id))

def sensor_listen(data, sensor_queue, sensor_id):
    sensor_queue.put((sensor_id, data.frame, data))
    return

def get_ego_id(recorder_file):
    found_lincoln = False
    found_id = None

    for line in recorder_file.split("\n"):

        # Check the role_name for hero
        if found_lincoln:
            if not line.startswith("  "):
                found_lincoln = False
                found_id = None
            else:
                data = line.split(" = ")
                if 'role_name' in data[0] and 'hero' in data[1]:
                    return found_id

        # Search for all lincoln vehicles
        if not found_lincoln and line.startswith(" Create ") and 'vehicle.lincoln' in line:
            found_lincoln = True
            found_id =  int(line.split(" ")[2][:-1])

    return found_id

def count_reward(map, hero):
    location = hero.get_location()
    lane_waypoint = map.get_waypoint(location, lane_type=carla.LaneType.Driving)
    EgoEnvState.error_dis = location.distance(lane_waypoint.transform.location)

    wp_yaw = lane_waypoint.transform.rotaion.yaw % 360
    actor_yaw = hero.get_transfrom().rotation.yaw % 360

    angle = (wp_yaw - actor_yaw) % 360

    angle = angle * np.pi / 180 if angle < 180 else (360 - angle) * np.pi / 180
    EgoEnvState.error_angle = abs(angle)
    
    EgoEnvState.speed = np.linalg.norm([hero.get_velocity().x, hero.get_velocity().y])
    control = hero.get_control()
    EgoEnvState.diff['steer'] = control.steer - EgoEnvState.steer
    EgoEnvState.diff['throttle'] = control.throttle - EgoEnvState.throttle

    EgoEnvState.steer = control.steer
    EgoEnvState.throttle, EgoEnvState.brake = control.throttle, control.brake

    if EgoEnvState.last_location is not None:
        diff_dis = location.distance(EgoEnvState.last_location)
        EgoEnvState.diff['complete'] = diff_dis * np.cos(angle)

    EgoEnvState.last_location = location

    return sum([v for v in step_reward_func().values()])

def save_data_to_disk(sensor_id, frame, data, imu_data, endpoint):
    """
    Saves the sensor data into file:
    - Images                        ->              '.png', one per frame, named as the frame id
    - Lidar:                        ->              '.ply', one per frame, named as the frame id
    - SemanticLidar:                ->              '.ply', one per frame, named as the frame id
    - RADAR:                        ->              '.csv', one per frame, named as the frame id
    - GNSS:                         ->              '.csv', one line per frame, named 'gnss_data.csv'
    - IMU:                          ->              '.csv', one line per frame, named 'imu_data.csv'
    """
    global CURRENT_THREADS
    CURRENT_THREADS += 1
    if isinstance(data, carla.Image):
        sensor_endpoint = f"{endpoint}/{sensor_id}/{frame}.png"
        data.save_to_disk(sensor_endpoint)

    elif isinstance(data, carla.LidarMeasurement):
        sensor_endpoint = f"{endpoint}/{sensor_id}/{frame}.npy"
        array_data = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1,4)
        np.save(sensor_endpoint, array_data)
        # data.save_to_disk(sensor_endpoint)

    # elif isinstance(data, carla.SemanticLidarMeasurement):
    #     sensor_endpoint = f"{endpoint}/{sensor_id}/{frame}.ply"
    #     data.save_to_disk(sensor_endpoint)

    # elif isinstance(data, carla.RadarMeasurement):
    #     sensor_endpoint = f"{endpoint}/{sensor_id}/{frame}.csv"
    #     data_txt = f"Altitude,Azimuth,Depth,Velocity\n"
    #     for point_data in data:
    #         data_txt += f"{point_data.altitude},{point_data.azimuth},{point_data.depth},{point_data.velocity}\n"
    #     with open(sensor_endpoint, 'w') as data_file:
    #         data_file.write(data_txt)

    elif isinstance(data, dict):
        measurements_file = f"{endpoint}/{sensor_id}/{frame}.json"
        f = open(measurements_file, "w")
        json.dump(data, f, indent=4)
        f.close()

    elif isinstance(data, carla.GnssMeasurement):
        pass
    #     sensor_endpoint = f"{endpoint}/{sensor_id}/gnss_data.csv"
    #     with open(sensor_endpoint, 'a') as data_file:
    #         data_txt = f"{frame},{data.altitude},{data.latitude},{data.longitude}\n"
    #         data_file.write(data_txt)

    elif isinstance(data, carla.IMUMeasurement):
        pass
    #     sensor_endpoint = f"{endpoint}/{sensor_id}/imu_data.csv"
    #     with open(sensor_endpoint, 'a') as data_file:
    #         data_txt = f"{frame},{imu_data[0][0]},{imu_data[0][1]},{imu_data[0][2]},{data.compass},{imu_data[1][0]},{imu_data[1][1]},{imu_data[1][2]}\n"
    #         data_file.write(data_txt)
    
    # elif isinstance(data, carla.VehicleControl):
    #     sensor_endpoint = f"{endpoint}/Control/control_data.csv"
    #     with open(sensor_endpoint, 'a') as data_file:
    #         data_txt = f"{frame},{data.throttle},{data.steer},{data.brake},{data.hand_brake},{data.reverse},{data.manual_gear_shift},{data.gear}\n"
    #         data_file.write(data_txt)

    else:
        print(f"WARNING: Ignoring sensor '{sensor_id}', as no callback method is known for data of type '{type(data)}'.")

    CURRENT_THREADS -= 1

def extract_imu_data(log):
    records = log["records"]
    log_data = []
    for record in records:
        acceleration_data = record["state"]["acceleration"]
        acceleration_vector = [acceleration_data["x"], acceleration_data["y"], acceleration_data["z"]]

        # TODO: Remove this (don't use logs without angular velocity)
        if "angular_velcoity" in record["state"]:
            angular_data = record["state"]["angular_velocity"]
            angular_vector = [angular_data["x"], angular_data["y"], angular_data["z"]]
        else:
            angular_vector = [random.random(), random.random(), random.random()]

        log_data.append([acceleration_vector, angular_vector])

    return log_data

def main():

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--host', default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument('--port', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    args = argparser.parse_args()
    print(__doc__)

    active_sensors = []

    try:

        # Initialize the simulation
        client = carla.Client(args.host, args.port)
        client.set_timeout(120.0)
        world = client.get_world()

        for recorder_info in RECORDER_INFO:
            recorder_folder = recorder_info['folder']
            recorder_start = recorder_info['start_time']
            recorder_duration = recorder_info['duration']

            # 0) Get the recorder files
            recorder_path_list = glob.glob(f"{os.getcwd()}/{recorder_folder}/*.log")
            if recorder_path_list:
                recorder_path = recorder_path_list[0]
            else:
                print(f"Couldn't find the recorder file for the folder '{recorder_folder}'. Stopping...")
                continue
            recorder_log_list = glob.glob(f"{os.getcwd()}/{recorder_folder}/log.json")
            if recorder_log_list:
                recorder_log = recorder_log_list[0]
            else:
                recorder_log = None

            print(f"\033[33m> Scenario name: {recorder_folder.split('/')[-1]}\033[0m")
            endpoint = f"{DESTINATION_FOLDER}/{recorder_path.split('/')[-1][:-4]}"

            # 1) Get the recorder map and load the world
            recorder_str = client.show_recorder_file_info(recorder_path, True)
            recorder_map = recorder_str.split("\n")[1][5:]
            print(f'> current map: {recorder_map}')
            world = client.load_world(recorder_map)
            world.tick()

            # 2) Change the weather and synchronous mode
            world.set_weather(WEATHER)
            settings = world.get_settings()
            settings.fixed_delta_seconds = 1 / FPS
            settings.synchronous_mode = True
            # settings.spectator_as_ego = False
            world.apply_settings(settings)

            for _ in range(100):
                world.tick()

            # 3) Replay the recorder
            max_duration = float(recorder_str.split("\n")[-2].split(" ")[1])
            if recorder_duration == 0:
                recorder_duration = max_duration
            elif recorder_start + recorder_duration > max_duration:
                print("WARNING: Found a duration that exceeds the recorder length. Reducing it from %s to %s"%(recorder_duration, max_duration - recorder_start))
                recorder_duration = max_duration - recorder_start
            if recorder_start >= max_duration:
                print("WARNING: Found a start point that exceeds the recoder duration. Ignoring it...")
                continue
            print(f"Duration: {round(recorder_duration, 2)} - Frames: {round(20*recorder_duration, 0)}")

            if recorder_log:
                with open(recorder_log) as fd:
                    log_json = json.load(fd)
                imu_logs = extract_imu_data(log_json)
            else:
                imu_logs = None

            client.replay_file(recorder_path, recorder_start, recorder_duration, get_ego_id(recorder_str), False)
            with open(f"{recorder_path[:-4]}.txt", 'w') as fd:
                fd.write(recorder_str)
            world.tick()

            # 4) Link onto the ego vehicle
            hero = None
            while hero is None:
                # print("Waiting for the ego vehicle...")
                possible_vehicles = world.get_actors().filter('vehicle.*')
                for vehicle in possible_vehicles:
                    if vehicle.attributes['role_name'] == 'hero':
                        # print("Ego vehicle found")
                        hero = vehicle
                        break
                time.sleep(5)

            world_map = world.get_map()

            # 5) Create the sensors, and save their data into a queue
            create_folders(endpoint, [[s.get('id'), s.get('type')] for s in SENSORS])
            blueprint_library = world.get_blueprint_library()
            sensor_queue = Queue()
            for sensor in SENSORS:

                # Extract the data from the sesor configuration
                sensor_id = sensor.get('id')
                attributes = sensor
                blueprint_name = attributes.get('type')
                sensor_transform = carla.Transform(
                    carla.Location(x=attributes.get('x'), y=attributes.get('y'), z=attributes.get('z')),
                    carla.Rotation(pitch=attributes.get('pitch'), roll=attributes.get('roll'), yaw=attributes.get('yaw'))
                )

                # Get the blueprint and add the attributes
                blueprint = blueprint_library.find(blueprint_name)
                # print(blueprint.attributes)
                for key, value in attributes.items():
                    if key in ['type', 'x', 'y', 'z', 'roll', 'pitch', 'yaw']:
                        continue
                    if key == "width":
                        blueprint.set_attribute("image_size_x", str(value))
                    elif key == "height":
                        blueprint.set_attribute("image_size_y", str(value))
                    elif blueprint.has_attribute(str(key)):
                        blueprint.set_attribute(str(key), str(value))


                # Create the sensors and its callback
                sensor = world.spawn_actor(blueprint, sensor_transform, hero)
                add_listener(sensor, sensor_queue, sensor_id)
                active_sensors.append(sensor)

            world.tick()

            # 6) Run the simulation
            start_time = world.get_snapshot().timestamp.elapsed_seconds
            start_frame = world.get_snapshot().frame
            sensor_amount = len(SENSORS)

            max_threads = THREADS
            results = []
            flag_init = True

            last_action = np.zeros(4)
            last_state = np.zeros(AgentConfig.state_dim)
            step = 0
            buffer_size = 1e6

            states = np.zeros([buffer_size, AgentConfig.state_dim])
            actions = np.zeros([buffer_size, 4])
            rewards = np.zeros(buffer_size)

            while True:
                current_time = world.get_snapshot().timestamp.elapsed_seconds
                current_duration = current_time - start_time
                if current_duration >= recorder_duration:
                    print(f">>>>>  Running recorded simulation: 100.00%  completed  <<<<<")
                    break

                completion = format(round(current_duration / recorder_duration * 100, 2), '3.2f')
                # print(f">>>>>  Running recorded simulation: {completion}%  completed  <<<<<", end="\r")

                # Get and save the sensor data from the queue.
                missing_sensors = sensor_amount
                sensor_data_dict = {}

                while True:

                    frame = world.get_snapshot().frame
                    try:
                        sensor_data = sensor_queue.get(True, 60.0)
                        if sensor_data[1] != frame:
                            continue  # Ignore previous frame data
                        missing_sensors -= 1
                    except Empty:
                        raise ValueError("A sensor took too long to send their data")

                    # Get the data
                    sensor_id = sensor_data[0]
                    frame_diff = sensor_data[1] - start_frame
                    data = sensor_data[2]
                    sensor_data_dict[sensor_id] = data
                    imu_data = data
                    # if imu_logs:
                    #     imu_data = imu_logs[int(FPS*recorder_start + frame_diff)]
                    # else:
                    #     imu_data = [[0,0,0], [0,0,0]]

                    # res = threading.Thread(target=save_data_to_disk, args=(sensor_id, frame_diff, data, imu_data, endpoint))
                    # results.append(res)
                    # res.start()

                    if CURRENT_THREADS > max_threads:
                        for res in results:
                            res.join()
                        results = []

                    if missing_sensors <= 0:
                        break

                world.tick()
                
                control_data = hero.get_control()
                speed = hero.get_velocity().squared_length()

                compass = imu_data.compass
                state = {}
                state["route"] = np.array([EgoEnvState.error_angle, EgoEnvState.error_dis])
                state["speed"] = np.ones(AgentConfig.speed_resample) * speed

                if abs(control_data.throttle) > abs(control_data.brake):
                    hero_action = np.array([control_data.steer, control_data.throttle])
                else:
                    hero_action = np.array([control_data.steer, -control_data.brake])

                state["last_action"] = np.repeat(last_action, AgentConfig.last_action_resample)
                last_action = hero_action

                bev = BEV_Generator(world, hero, compass, command=4) # 4 for lane-follow
                state['bev'] = bev.step().reshape(-1)

                states[step] = np.concatenate([feature for feature in state.values()])
                actions[step] = hero_action
                rewards[step] = count_reward()
                step += 1

            for res in results:
                res.join()
            
            print('data extraction done, total length:', step)
            length = min(step, buffer_size)

            np.save(f'{endpoint}/states.npy', states[:length])
            np.save(f'{endpoint}/actions.npy', actions[:length])
            np.save(f'{endpoint}/rewards.npy', rewards[:length])

            for sensor in active_sensors:
                sensor.stop()
                sensor.destroy()
            active_sensors = []

            for _ in range(50):
                world.tick()

    # End the simulation
    finally:
        # stop and remove cameras
        for sensor in active_sensors:
            sensor.stop()
            sensor.destroy()

        # set fixed time step length
        settings = world.get_settings()
        settings.fixed_delta_seconds = None
        settings.synchronous_mode = False
        world.apply_settings(settings)

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
