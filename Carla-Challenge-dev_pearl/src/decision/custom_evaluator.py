#!/usr/bin/env python
# Copyright (c) 2018-2019 Intel Corporation.
# authors: German Ros (german.ros@intel.com), Felipe Codevilla (felipe.alcm@gmail.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
CARLA Challenge Evaluator Routes

Provisional code to evaluate Autonomous Agents for the CARLA Autonomous Driving challenge
"""
from __future__ import print_function

import traceback
import argparse
from argparse import RawTextHelpFormatter
import importlib
import os
import sys
import time
import signal

import carla
from srunner.scenariomanager.carla_data_provider import *
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog
from leaderboard.envs.sensor_interface import SensorConfigurationInvalid
from leaderboard.autoagents.agent_wrapper import AgentError, validate_sensor_configuration
from leaderboard.utils.statistics_manager import StatisticsManager, FAILURE_MESSAGES

from custom_leaderboard.scenarios.scenario_manager import ScenarioManager
from custom_leaderboard.utils.route_indexer import RouteIndexer
from custom_leaderboard.scenarios.route_scenario import RouteScenario

from utils.config_parser import AgentConfig


class LeaderboardEvaluator(object):
    """
    Main class of the Leaderboard. Everything is handled from here,
    from parsing the given files, to preparing the simulation, to running the route.
    """

    # Tunable parameters
    client_timeout = 10.0  # in seconds
    frame_rate = 20.0  # in Hz

    def __init__(self, args, statistics_manager):
        """
        Setup CARLA client and world
        Setup ScenarioManager
        """
        self.world = None
        self.manager = None
        self.sensors = None
        self.sensors_initialized = False
        self.sensor_icons = []
        self.agent_instance = None
        self.route_scenario = None

        self.statistics_manager = statistics_manager
        self.sensors_to_icons = {
            "sensor.camera.rgb": "carla_camera",
            "sensor.camera.semantic_segmentation": "carla_camera",
            "sensor.camera.depth": "carla_camera",
            "sensor.lidar.ray_cast": "carla_lidar",
            "sensor.lidar.ray_cast_semantic": "carla_lidar",
            "sensor.other.radar": "carla_radar",
            "sensor.other.gnss": "carla_gnss",
            "sensor.other.imu": "carla_imu",
            "sensor.opendrive_map": "carla_opendrive_map",
            "sensor.speedometer": "carla_speedometer",
        }

        # This is the ROS1 bridge server instance. This is not encapsulated inside the ROS1 agent because the same
        # instance is used on all the routes (i.e., the server is not restarted between routes). This is done
        # to avoid reconnection issues between the server and the roslibpy client.
        self._ros1_server = None

        # Setup the simulation
        self.client, self.client_timeout, self.traffic_manager = self._setup_simulation(args)

        # dist = pkg_resources.get_distribution("carla")
        # if dist.version != 'leaderboard':
        #     if LooseVersion(dist.version) < LooseVersion('0.9.10'):
        #         raise ImportError("CARLA version 0.9.10.1 or newer required. CARLA version found: {}".format(dist))

        # Load agent
        module_name = os.path.basename(args.agent).split(".")[0]
        sys.path.insert(0, os.path.dirname(args.agent))
        self.module_agent = importlib.import_module(module_name)

        # Create the ScenarioManager
        self.manager = ScenarioManager(args.timeout, self.statistics_manager, args.debug)

        # Time control for summary purposes
        self._start_time = GameTime.get_time()
        self._end_time = None

        # Prepare the agent timer
        self._agent_watchdog = None
        signal.signal(signal.SIGINT, self._signal_handler)

        self._client_timed_out = False
        self._route_rep_cnt = 0

    def _signal_handler(self, signum, frame):
        """
        Terminate scenario ticking when receiving a signal interrupt.
        Either the agent initialization watchdog is triggered, or the runtime one at scenario manager
        """
        if self._agent_watchdog and not self._agent_watchdog.get_status():
            raise RuntimeError(
                "Timeout: Agent took longer than {}s to setup".format(self.client_timeout)
            )
        elif self.manager:
            self.manager.signal_handler(signum, frame)

    def __del__(self):
        """
        Cleanup and delete actors, ScenarioManager and CARLA world
        """
        if hasattr(self, "manager") and self.manager:
            del self.manager
        if hasattr(self, "world") and self.world:
            del self.world

    def _get_running_status(self):
        """
        returns:
           bool: False if watchdog exception occured, True otherwise
        """
        if self._agent_watchdog:
            return self._agent_watchdog.get_status()
        return False

    def _cleanup(self, reload=True):
        """
        Remove and destroy all actors
        """

        if reload:
            CarlaDataProvider.cleanup()

            if self.route_scenario:
                self.route_scenario.remove_all_actors()
                self.route_scenario = None

            if self._agent_watchdog:
                self._agent_watchdog.stop()

            if self.statistics_manager:
                self.statistics_manager.remove_scenario()

            try:
                if self.agent_instance:
                    self.agent_instance.destroy()
                    self.agent_instance = None
            except Exception as e:
                print("\n\033[91mFailed to stop the agent:")
                print(f"\n{traceback.format_exc()}\033[0m")

            # Make sure no sensors are left streaming
            alive_sensors = self.world.get_actors().filter("*sensor*")
            for sensor in alive_sensors:
                sensor.stop()
                sensor.destroy()

        else:
            # reset CarlaDataProvider
            initial_action = carla.VehicleControl(brake=1.0, hand_brake=True)
            self.manager.ego_vehicles[0].apply_control(initial_action)
            for i in range(8):
                self.world.tick()

            DestroyActor = carla.command.DestroyActor
            ApplyTransform = carla.command.ApplyTransform

            CarlaDataProvider.set_runtime_init_mode(False)

            batch = []

            for actor_id in CarlaDataProvider._carla_actor_pool.copy():
                actor = CarlaDataProvider._carla_actor_pool[actor_id]
                if actor is not None and actor.is_alive:
                    if actor.attributes["role_name"] == "hero":
                        hero_transf = self._backup["transf_map"][actor]
                        batch.append(ApplyTransform(actor, hero_transf))
                    else:
                        batch.append(DestroyActor(actor))
                        CarlaDataProvider._carla_actor_pool[actor_id] = None
                        CarlaDataProvider._carla_actor_pool.pop(actor_id)

            if CarlaDataProvider._client:
                try:
                    CarlaDataProvider._client.apply_batch_sync(batch)
                except RuntimeError as e:
                    if "time-out" in str(e):
                        pass
                    else:
                        raise e

            CarlaDataProvider.on_carla_tick()
            self.world.tick()
            self.route_scenario.reset()

        if self.manager:
            self._client_timed_out = not self.manager.get_running_status()
            self.manager.cleanup(reload)


    def _get_initial_states(self):
        def copymap(_map):
            _memory = {}
            for actor, value in _map.items():
                _memory[actor] = value
            return _memory

        # print(self.route_scenario.__dict__)
        # print(self.route_scenario.__dir__())

        self._backup = {
            "transf_map": copymap(CarlaDataProvider._actor_transform_map),
            "loc_map": copymap(CarlaDataProvider._actor_location_map),
            "vel_map": copymap(CarlaDataProvider._actor_velocity_map),
        }

    def _setup_simulation(self, args):
        """
        Prepares the simulation by getting the client, and setting up the world and traffic manager settings
        """
        client = carla.Client(args.host, args.port)
        if args.timeout:
            client_timeout = args.timeout
        client.set_timeout(client_timeout)

        settings = carla.WorldSettings(
            synchronous_mode=True,
            fixed_delta_seconds=1.0 / self.frame_rate,
            deterministic_ragdolls=True,
            spectator_as_ego=False,
        )
        client.get_world().apply_settings(settings)

        traffic_manager = client.get_trafficmanager(args.traffic_manager_port)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_hybrid_physics_mode(True)

        return client, client_timeout, traffic_manager

    def _reset_world_settings(self):
        """
        Changes the modified world settings back to asynchronous
        """
        # Has simulation failed?
        if self.world and self.manager and not self._client_timed_out:
            # Reset to asynchronous mode
            self.world.tick()  # TODO: Make sure all scenario actors have been destroyed
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            settings.deterministic_ragdolls = False
            settings.spectator_as_ego = True
            self.world.apply_settings(settings)

            # Make the TM back to async
            self.traffic_manager.set_synchronous_mode(False)
            self.traffic_manager.set_hybrid_physics_mode(False)

    def _load_and_wait_for_world(self, args, town):
        """
        Load a new CARLA world without changing the settings and provide data to CarlaDataProvider
        """

        self.world = self.client.load_world(town, reset_settings=False)

        # Large Map settings are always reset, for some reason
        settings = self.world.get_settings()
        settings.tile_stream_distance = 650
        settings.actor_active_distance = 650
        self.world.apply_settings(settings)
        self.world.reset_all_traffic_lights()

        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_traffic_manager_port(args.traffic_manager_port)
        CarlaDataProvider.set_world(self.world)

        # This must be here so that all route repetitions use the same 'unmodified' seed
        self.traffic_manager.set_random_device_seed(args.traffic_manager_seed)

        # Wait for the world to be ready
        self.world.tick()

        map_name = CarlaDataProvider.get_map().name.split("/")[-1]
        if map_name != town:
            raise Exception(
                "The CARLA server uses the wrong map!"
                " This scenario requires the use of map {}".format(town)
            )

    def _register_statistics(self, route_index, entry_status, crash_message=""):
        """
        Computes and saves the route statistics
        """
        print("\033[1m> Registering the route statistics\033[0m")
        self.statistics_manager.save_entry_status(entry_status)
        route_status = self.statistics_manager.compute_route_statistics(
            route_index,
            self.manager.scenario_duration_system,
            self.manager.scenario_duration_game,
            crash_message,
        )
        return route_status

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
        self.agent_instance.setup(args.agent_config)

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
    
    def _load_and_run_scenario(self, args, config, next_route, initialize):
        """
        Load and run the scenario given by config.

        Depending on what code fails, the simulation will either stop the route and
        continue from the next one, or report a crash and stop.
        """
        crash_message = ""
        entry_status = "Started"

        print(
            "\n\033[1m===== Preparing {} (episode {}) ====\033[0m".format(
                config.name, self._route_rep_cnt
            )
        )
        AgentConfig.parse_config(args.agent_config)
        AgentConfig.set_scenario(args.scenario_name)

        # Prepare the statistics of the route
        route_name = f"{config.name}_rep{self._route_rep_cnt}"
        self.statistics_manager.create_route_data(route_name, config.index)

        # Load the world and the scenario
        try:
            start = time.time()
            if next_route or initialize:
                print("\033[1m> Loading the world\033[0m")
                self._route_rep_cnt = 0
                self._load_and_wait_for_world(args, config.town)
                self.route_scenario = RouteScenario(
                    scenario_name=args.scenario_name,
                    world=self.world, config=config, debug_mode=args.debug
                )
                self.statistics_manager.set_scenario(self.route_scenario)
            else:
                print("\033[1m> Retry the previous route\033[0m")
                # self.route_scenario = RouteScenario(world=self.world, config=config, debug_mode=args.debug)
                # self.statistics_manager.set_scenario(self.route_scenario)
                self._route_rep_cnt += 1

        except Exception:
            # The scenario is wrong -> set the ejecution to crashed and stop
            print("\n\033[91mThe scenario could not be loaded:")
            print(f"\n{traceback.format_exc()}\033[0m")

            entry_status, crash_message = FAILURE_MESSAGES["Simulation"]
            self._register_statistics(config.index, entry_status, crash_message)
            self._cleanup()
            return True, False

        # Set up the user's agent, and the timer to avoid freezing the simulation
        try:
            if next_route or initialize:
                self._agent_watchdog = Watchdog(args.timeout)
                self._agent_watchdog.start()

                self._setup_agent(args)

                self._agent_watchdog.stop()
                self._agent_watchdog = None
            else:
                self.agent_instance.reset()

        except SensorConfigurationInvalid as e:
            # The sensors are invalid -> set the ejecution to rejected and stop
            print("\n\033[91mThe sensor's configuration used is invalid:")
            print(f"{e}\033[0m\n")

            entry_status, crash_message = FAILURE_MESSAGES["Sensors"]
            self._register_statistics(config.index, entry_status, crash_message)
            self._cleanup()
            return True, False

        except Exception:
            # The agent setup has failed -> start the next route
            print("\n\033[91mCould not set up the required agent:")
            print(f"\n{traceback.format_exc()}\033[0m")

            entry_status, crash_message = FAILURE_MESSAGES["Agent_init"]
            self._register_statistics(config.index, entry_status, crash_message)
            self._cleanup()
            return True, False

        print("\033[1m> Running the route\033[0m")

        # Run the scenario
        try:
            # Load scenario and run it
            if args.record:
                self.client.start_recorder(
                    "{}/{}_rep{}.log".format(args.record, config.name, self._route_rep_cnt)
                )

            if next_route or initialize:
                self.manager.load_scenario(
                    self.route_scenario, self.agent_instance, config.index, config.repetition_index
                )
                self.manager.run_scenario(reload=True)
                # tick once before getting initial states for the fucking scenarios like ParkingExit
                self.manager._tick_scenario()
                self._get_initial_states()
                print(f"\033[33m- Reload time: {time.time()-start}\033[0m")
            else:
                self.manager.run_scenario(reload=False)

            # loop for simulation
            while self.manager._running:
                self.manager._tick_scenario()

        except AgentError:
            # The agent has failed -> stop the route
            print("\n\033[91mStopping the route, the agent has crashed:")
            print(f"\n{traceback.format_exc()}\033[0m")
            print("an error occur, exiting...")
            self.manager._running = False
            self.manager.stop_scenario()
            self._cleanup()
            entry_status, crash_message = FAILURE_MESSAGES["Agent_runtime"]
            return False, False

        except Exception:
            print("\n\033[91mError during the simulation:")
            print(f"\n{traceback.format_exc()}\033[0m")
            print("an error occur, exiting...")
            self.manager._running = False
            self.manager.stop_scenario()
            self._cleanup()

            entry_status, crash_message = FAILURE_MESSAGES["Simulation"]
            return False, False

        # Stop the scenario
        try:
            start = time.time()
            print("\033[1m> Stopping the route\033[0m")
            self.manager.stop_scenario(next_route)
            # register statistics and get route status: "Failed", "Completed" or "Perfect"
            # route_status = self._register_statistics(config.index, entry_status, crash_message)
            # print('> config.index', config.index)
            route_status = self._register_statistics(0, entry_status, crash_message)
            print(f"\033[1m> route status: {route_status}\033[0m")

            if args.record:
                self.client.stop_recorder()

            self._cleanup(next_route)
            print(f"\033[33m- Reset time: {time.time()-start}\033[0m")

        except Exception:
            print("\n\033[91mFailed to stop the scenario, the statistics might be empty:")
            print(f"\n{traceback.format_exc()}\033[0m")

            _, crash_message = FAILURE_MESSAGES["Simulation"]

        # If the simulation crashed, stop the leaderboard, for the rest,
        # move to the next route if get perfect score in the current route, else repete
        return crash_message == "Simulation crashed", route_status == "Perfect"

    def run(self, args):
        """
        Run the challenge mode
        """
        route_indexer = RouteIndexer(args.routes, args.repetitions, args.routes_subset)

        if args.resume:
            resume = route_indexer.validate_and_resume(args.checkpoint)
        else:
            resume = False

        if resume:
            self.statistics_manager.add_file_records(args.checkpoint)
        else:
            self.statistics_manager.clear_records()
        self.statistics_manager.save_progress(route_indexer.index, route_indexer.total)
        self.statistics_manager.write_statistics()

        # crashed: simulation crashed, accident of agent is not included
        crashed, next_route, initialize = False, False, True
        while route_indexer.peek() and not crashed:
            # Run the scenario
            config = route_indexer.get_next_config(next_route)
            crashed, next_route = self._load_and_run_scenario(
                args, config, next_route, initialize
            )
            initialize = False

            # Save the progress and write the route statistics
            self.statistics_manager.save_progress(route_indexer.index, route_indexer.total)
            self.statistics_manager.write_statistics()

        # Shutdown ROS1 bridge server if necessary
        if self._ros1_server is not None:
            self._ros1_server.shutdown()

        # Go back to asynchronous mode
        self._reset_world_settings()

        if not crashed:
            # Save global statistics
            print("\033[1m> Registering the global statistics\033[0m")
            self.statistics_manager.compute_global_statistics()
            self.statistics_manager.validate_and_write_statistics(self.sensors_initialized, crashed)

        return crashed


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
        "--timeout", default=300.0, type=float, help="Set the CARLA client timeout value in seconds"
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
        "--scenario-name",
        type=str,
        default=None,
        help="Specific scenario to train"
    )

    arguments = parser.parse_args()

    statistics_manager = StatisticsManager(arguments.checkpoint, arguments.debug_checkpoint)
    leaderboard_evaluator = LeaderboardEvaluator(arguments, statistics_manager)
    crashed = leaderboard_evaluator.run(arguments)

    del leaderboard_evaluator

    if crashed:
        sys.exit(-1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
