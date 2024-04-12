import py_trees
from utils.config_parser import AgentConfig
from srunner.scenariomanager.timer import GameTime


REWARD_TERMINAL = {
    "OutsideRouteLanesTest": -5,
    "CollisionTest": -5,
    "RunningRedLightTest": -5,
    "RunningStopTest": -5,
    "MinimumSpeedRouteTest": -5,
    "InRouteTest": -5,
    "ActorBlockedTest": -5,
}

def step_reward_func():
    reward = {}

    dis = round(EgoEnvState.error_dis, 3)
    angle = round(EgoEnvState.error_angle, 3)
    ego_speed = round(EgoEnvState.speed, 3)
    reward['angle'] = 0
    reward['distance'] = 0
    reward['avg_speed'] = 0

    if EgoEnvState.route_tick:
        if AgentConfig.action_mask_type != 1:
            ## not Junction scenario
            if dis < 1 and ego_speed > 0.5:
                reward['distance'] = 0.4*(1-dis)
                if abs(angle) < 0.2:
                    reward['angle'] += 0.2 - abs(angle)
            if dis > 1:
                reward['distance'] = max(-0.2, -0.1*(dis-1))

        reward['avg_speed'] = 0.5 / EgoEnvState.step if EgoEnvState.step != 0 else 0.5
        EgoEnvState.step = 0

    else:
        EgoEnvState.step += 1

    # if EgoEnvState.route_completion > 10:
    # print('showing complete:', EgoEnvState.route_completion)
        
    return reward

def calculate_terminal_reward():
    reward = 5 * EgoEnvState.route_completion / AgentConfig.exp_config_dict['single_scence_length']
    return reward

# def reward_collision(obj):
#     speed = getattr(obj, "actor_velocity")
#     return -20

def reward_complete(obj):
    finish_time = getattr(obj, "step") / 1000

    reward = 5
    reward += max(0, 5 * (1 - finish_time))

    if AgentConfig.action_mask_type == 1:
        reward -= 5 *abs(EgoEnvState.route_aim)

    EgoEnvState.finish_time = finish_time
    return reward

class EgoEnvState:
    step = 0
    error_dis = 0
    error_angle = 0
    route_aim = 0
    speed = 0
    steer = 0
    throttle = 0
    brake = 0
    finish_time = -1
    route_tick = False
    road_speed_limit = 0
    route_completion = 0
    diff = {"steer": 0 , "throttle": 0, "complete": 0}
    last_location = None

def getstate_transf(obj):
    EgoEnvState.error_dis = getattr(obj, "distance")
    EgoEnvState.error_angle = getattr(obj, "angle")

def getstate_control(obj):
    EgoEnvState.brake = getattr(obj, "actor_brake") 
    EgoEnvState.steer = getattr(obj, "actor_steer")
    EgoEnvState.throttle = getattr(obj, "actor_throttle")
    EgoEnvState.diff['steer'] = getattr(obj, "steer_diff")
    EgoEnvState.diff['throttle'] = getattr(obj, "throttle_diff")
    EgoEnvState.speed = getattr(obj, "actor_velocity")
    EgoEnvState.road_speed_limit = getattr(obj, "speed_limit")

def getstate_rc(obj):
    EgoEnvState.diff['complete'] = getattr(obj, "meter_diff")
    EgoEnvState.route_tick = getattr(obj, 'route_tick')
    EgoEnvState.route_completion = getattr(obj, 'actual_meter')


# To reduce complexity, calculate step reward when checking criteria
CHECK_STATE_DICT = {
    "OutsideRouteLanesTest": getstate_transf,
    "MinimumSpeedRouteTest": getstate_control,
    "RouteCompletionTest": getstate_rc,
}

class RewardCounter:
    reward = {'Terminal': 0}
    terminal = False
    count = 0

    @classmethod
    def wrapper(cls, func):
        criteria = func.__qualname__.split(".")[0]

        def func_with_reward_calculated(*args, **kwargs):
            new_status = func(*args, **kwargs)
            ## calculate terminal reward
            if new_status == py_trees.common.Status.FAILURE:
                cls.terminal = True
                # reset test status to running mode
                setattr(args[0], "test_status", "SUCCESS")
                # if criteria == "CollisionTest":
                #     cls.reward['Terminal'] += reward_collision(args[0])
                if criteria == "RouteCompletionTest":
                    cls.reward['Terminal'] += reward_complete(args[0])
                elif criteria in REWARD_TERMINAL.keys():
                    cls.reward['Terminal'] += REWARD_TERMINAL[criteria]
                    cls.reward['Terminal'] += calculate_terminal_reward()

                print(f"\033[33m> {criteria} falied, which may lead to reset\033[0m")

            ## get vars for step reward calculation
            if criteria in CHECK_STATE_DICT.keys():
                CHECK_STATE_DICT[criteria](args[0])
            return new_status

        return func_with_reward_calculated

    @classmethod
    def reset(cls):
        cls.terminal = False
        for key in cls.reward:
            cls.reward[key] = 0

    @classmethod
    def step(cls):
        terminal = cls.terminal
        reward = {**cls.reward, **step_reward_func()}
        cls.count += 1
        if cls.count % 20 == 0:
            cls.count = 0
            print("> showing reward: ", reward, terminal)
        cls.reset()
        return reward, terminal
