import torch
from torch import nn
import numpy as np

from pearl.neural_networks.common.utils import mlp_block, conv_block
from pearl.neural_networks.sequential_decision_making.q_value_networks import VanillaQValueNetwork
from pearl.neural_networks.sequential_decision_making.actor_networks import (
    DynamicActionActorNetwork, GaussianActorNetwork
)
from pearl.replay_buffers.sequential_decision_making.fifo_off_policy_replay_buffer import (
    FIFOOffPolicyReplayBuffer,
)
from pearl.api.state import SubjectiveState


class MultiStateReplayBuffer(FIFOOffPolicyReplayBuffer):
    '''
    MultiStateRB receive state in form of dict containing nparrays,
    flatten then into 1-D and plug it in TensorBasedReplayBuffer
    States in the buffer should be reshaped in corresponding network
    '''
    def __init__(self, capacity: int, has_cost_available: bool = False) -> None:
        super().__init__(capacity, has_cost_available)
        SubjectiveState = dict

    def _process_single_state(self, state: SubjectiveState) -> torch.Tensor:
        # convert multi state to 1-D nparray
        state = np.concatenate([feature.reshape(-1) for feature in state.values()])
        # convert nparray to tensor
        return super()._process_single_state(state)
    

def recover_state_from_buffer(state: torch.Tensor):
    from utils.config_parser import AgentConfig
    ms_len = AgentConfig.measurement_dim
    measurement, bev = torch.split(state,
                                   [ms_len, state.shape[-1] - ms_len],
                                   dim=-1)
    _shape = measurement.shape[:-1]
    # complex reshape to promise the data in bev feature having order as before
    # not using .permute() because discretized SAC in pearl will pass 
    # both 3D and 2D state to actor/critic net 
    bev = bev.reshape(*_shape, 20, 20, 7).transpose(-2,-1).transpose(-3,-2)

    return measurement, bev.reshape(-1, 7, 20, 20)

def get_multi_head():
    bev_head = nn.Sequential(
        conv_block( input_channels_count=7,
                    output_channels_list=[14, 28],
                    kernel_sizes=[5, 5],
                    paddings=[0,0],
                    strides=[3,1]),  # output:2*2*28
        nn.Flatten(start_dim=-2)
    )
    from utils.config_parser import AgentConfig
    ms_len = AgentConfig.measurement_dim
    measurement_head = mlp_block(input_dim=ms_len,
                                 hidden_dims=[128, 64],
                                 output_dim=64)
    
    return bev_head, measurement_head

    
class MultiStateCritic(VanillaQValueNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bev_head, self._measure_head = get_multi_head()
        
    def get_q_values(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:       
        measurement, bev = recover_state_from_buffer(state)
        x1 = self._measure_head(measurement)
        x2 = self._bev_head(bev).reshape(*x1.shape[:-1], -1)
        state = torch.cat([x1, x2], dim=-1)
        return super().get_q_values(state, action)
    
class MultiStateActor(DynamicActionActorNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bev_head, self._measure_head = get_multi_head()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state, action = torch.split(x,
                                    [x.shape[-1]-2, 2],
                                    dim=-1)
        measurement, bev = recover_state_from_buffer(state)    
        x1 = self._measure_head(measurement)
        x2 = self._bev_head(bev).reshape(*x1.shape[:-1], -1)

        x = torch.cat([x1, x2, action], dim=-1)
        return super().forward(x)
    
class MultiStateGaussianActor(GaussianActorNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bev_head, self._measure_head = get_multi_head()

    def forward(self, state: torch.Tensor):
        measurement, bev = recover_state_from_buffer(state)
        x1 = self._measure_head(measurement)
        x2 = self._bev_head(bev).reshape(*x1.shape[:-1], -1)
        state = torch.cat([x1, x2], dim=-1)
        return super().forward(state)

    
    


# agent = PearlAgent(
#     policy_learner=DeepQLearning(
#         state_dim=env.observation_space.shape[0],
#         action_space=env.action_space,
#         hidden_dims=[64, 64],
#         training_rounds=20,
#         action_representation_module=OneHotActionTensorRepresentationModule(
#             max_number_actions=num_actions
#         ),
#     ),
#     replay_buffer=FIFOOffPolicyReplayBuffer(10_000),
# )

# observation, action_space = env.reset()
# agent.reset(observation, action_space)
# done = False
# while not done:
#     action = agent.act(exploit=False)
#     action_result = env.step(action)
#     agent.observe(action_result)
#     agent.learn()
#     done = action_result.done