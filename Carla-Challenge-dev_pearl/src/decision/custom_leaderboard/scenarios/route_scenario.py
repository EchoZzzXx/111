#!/usr/bin/env python

# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides Challenge routes as standalone scenarios
"""

from __future__ import print_function

import py_trees
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from envlib.atomic_criteria import (
    CollisionTest,
    InRouteTest,
    RouteCompletionTest,
    OutsideRouteLanesTest,
    RunningRedLightTest,
    RunningStopTest,
    ActorBlockedTest,
    MinimumSpeedRouteTest,
)

from leaderboard.scenarios import route_scenario
from srunner.scenarioconfigs.scenario_configuration import ActorConfigurationData


class RouteScenario(route_scenario.RouteScenario):
    def __init__(self, scenario_name, *args, **kwargs):
        self.specific_scenario = scenario_name
        super().__init__(*args, **kwargs)

    def _spawn_ego_vehicle(self):
        """Spawn the ego vehicle at the first waypoint of the route"""
        elevate_transform = self.route[0][0]
        elevate_transform.location.z += 0.5

        ego_vehicle = CarlaDataProvider.request_new_actor(
            "vehicle.lincoln.mkz_2020", elevate_transform, rolename="hero"
        )
        if not ego_vehicle:
            return

        spectator = self.world.get_spectator()
        spectator.set_transform(
            carla.Transform(
                elevate_transform.location + carla.Location(z=30), carla.Rotation(pitch=-90)
            )
        )

        self.world.tick()

        return ego_vehicle
    
    def _get_route(self, config):
        '''
        set the startpoint of the route to specific scenario
        '''
        interpolated_route = super()._get_route(config)
        if self.specific_scenario is None:
            return interpolated_route
        
        ego_spawn_ref = None
        scenario_distance = None
        for sconfig in self.config.scenario_configs:
            if sconfig.type == self.specific_scenario:
                ego_spawn_ref = sconfig.trigger_points[0]
                if 'distance' in sconfig.other_parameters:
                    _str = sconfig.other_parameters['distance']['value']
                    scenario_distance = int(_str)
        assert ego_spawn_ref is not None

        dis_threshold = 5
        _index = 0
        for i, guideline in enumerate(interpolated_route):
            keypoint, command = guideline
            dis = keypoint.location.distance(ego_spawn_ref.location)
            if dis < dis_threshold:                
                _index = i
                break

        print('> showing forward distance:', dis)
                
        from utils.config_parser import AgentConfig
        forward_distance = AgentConfig.exp_config_dict['spawn_forward']
        if forward_distance is not None:
            _sign = 1
            if forward_distance < 0:
                _sign = -1
            while dis < _sign *forward_distance:
                _index += 1 *_sign
                keypoint, _ = interpolated_route[_index]
                dis = keypoint.location.distance(ego_spawn_ref.location)
            print('> showing dis and forward:', dis, forward_distance)
            return interpolated_route[_index:]

        if _index <= 1:
            return interpolated_route

        ## set the startpoint of the route into specific scenario
        from leaderboard.utils import route_parser
        
        scenario_distance = {"EnterActorFlowV2": 100,
                             "NonSignalizedJunctionRightTurn": 50,
                             "OppositeVehicleRunningRedLight": 120}[self.specific_scenario] \
        if scenario_distance is None else scenario_distance

        setattr(route_parser, 'DIST_THRESHOLD', scenario_distance)
        setattr(route_parser, 'ANGLE_THRESHOLD', 30)

        while dis < scenario_distance / 2 + 15:
            _index += 1
            keypoint, _ = interpolated_route[_index]
            dis = keypoint.location.distance(ego_spawn_ref.location)

        return interpolated_route[_index:]


    def reset(self):
        self.other_actors = []
        self.list_scenarios = []
        self.occupied_parking_locations = []
        self.available_parking_locations = []

        self.behavior_tree.remove_child(self.behavior_node)
        self.criteria_tree.remove_all_children()
        self.criteria_tree.status = py_trees.common.Status.RUNNING
        self.behavior_tree.add_child(self._create_behavior())
        self.criteria_tree.add_child(self._create_test_criteria())

        self.scenario_tree.remove_child(self.timeout_node)
        self.timeout_node = self._create_timeout_behavior()
        self.scenario_tree.add_child(self.timeout_node)

        self.scenario_tree.status = py_trees.common.Status.RUNNING
        self.scenario_tree.setup(timeout=1)
        self.scenario_tree.initialise()

        self.missing_scenario_configurations = self.scenario_configurations.copy()
        self._parked_ids = []
        self._get_parking_slots()


    def _create_test_criteria(self):
        """
        Create the criteria tree. It starts with some route criteria (which are always active),
        and adds the scenario specific ones, which will only be active during their scenario
        """
        criteria = py_trees.composites.Parallel(
            name="Criteria", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE
        )

        self.criteria_node = criteria

        # End condition
        criteria.add_child(RouteCompletionTest(self.ego_vehicles[0], route=self.route, terminate_on_failure=True))

        # 'Normal' criteria
        criteria.add_child(OutsideRouteLanesTest(self.ego_vehicles[0], route=self.route, terminate_on_failure=True))
        criteria.add_child(
            CollisionTest(self.ego_vehicles[0], name="CollisionTest", terminate_on_failure=True)
        )
        criteria.add_child(RunningRedLightTest(self.ego_vehicles[0], terminate_on_failure=False))
        criteria.add_child(RunningStopTest(self.ego_vehicles[0], terminate_on_failure=True))
        criteria.add_child(
            MinimumSpeedRouteTest(
                self.ego_vehicles[0],
                self.route,
                checkpoints=4,
                name="MinSpeedTest",
                terminate_on_failure=True,
            )
        )

        # These stop the route early to save computational time
        criteria.add_child(
            InRouteTest(
                self.ego_vehicles[0], route=self.route, offroad_max=30, terminate_on_failure=True
            )
        )
        criteria.add_child(
            ActorBlockedTest(
                self.ego_vehicles[0],
                min_speed=1,
                max_time=5,
                terminate_on_failure=True,
                name="AgentBlockedTest",
            )
        )

        return criteria
    