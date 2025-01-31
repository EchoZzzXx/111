# export WORKSPACE=/home/rowena/CARLA2023
# export CARLA_ROOT=${WORKSPACE}/../CARLA2023/CARLA
# export LEADERBOARD_ROOT=${WORKSPACE}/leaderboard
# export SCENARIO_RUNNER_ROOT=${WORKSPACE}/scenario_runner
# # export TEAM_CODE_ROOT=${WORKSPACE}/teamcode

# export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
# export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
# export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.14-py3.7-linux-x86_64.egg
# # export PYTHONPATH=$PYTHONPATH:${TEAM_CODE_ROOT}/data

export WORKSPACE=${PWD}
export CARLA_ROOT=${WORKSPACE}/../CARLA2023/CARLA
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh

export THIRD_PARTY_ROOT=${WORKSPACE}/third-party
export LEADERBOARD_ROOT=${THIRD_PARTY_ROOT}/leaderboard
export SCENARIO_RUNNER_ROOT=${THIRD_PARTY_ROOT}/scenario_runner
export RLLIB_ROOT=${THIRD_PARTY_ROOT}/rllib

export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.14-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:${LEADERBOARD_ROOT}
export PYTHONPATH=$PYTHONPATH:${SCENARIO_RUNNER_ROOT}
export PYTHONPATH=$PYTHONPATH:${RLLIB_ROOT}
export PYTHONPATH=$PYTHONPATH:'./src/decision'

cd ./src/decision
python3 ./data_extractor.py