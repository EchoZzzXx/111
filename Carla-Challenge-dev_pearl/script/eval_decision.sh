#!/bin/bash
source activate carla-rl;
export WORKSPACE=${PWD}
export CARLA_ROOT=${WORKSPACE}/../CARLA
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

export CHALLENGE_TRACK_CODENAME=SENSORS
export PORT=2005
export TM_PORT=2505
export DEBUG_CHALLENGE=0
export REPETITIONS=1 # multiple evaluation runs
export RESUME=0
# export TEAM_CONFIG=leaderboard/team_code/interfuser_config.py # model checkpoint, not required for expert
# export SAVE_PATH=data/eval # path for saving episodes while evaluating

kill -9 `lsof -t -i:${PORT}`
nohup ${CARLA_SERVER} -carla-rpc-port=$PORT -windowed -ResX=400 -ResY=300 &
sleep 15

# kill the process of previous traffic manager to avoid blind error
kill -9 `lsof -t -i:${TM_PORT}`
cd ./src/decision

python3 ./custom_evaluator.py \
--routes=${LEADERBOARD_ROOT}/data/routes_training.xml \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--agent=${WORKSPACE}/src/decision/agent/agent_single.py \
--agent-config=${WORKSPACE}/src/decision/configs/config_single.yaml \
--debug=${DEBUG_CHALLENGE} \
--resume=${RESUME} \
--routes-subset="15" \
--port=${PORT} \
--traffic-manager-port=${TM_PORT} \
--scenario-name=EnterActorFlowV2

## scenario list
# - ParkingExit
# - Accident
# - OppositeVehicleRunningRedLight
# - DynamicObjectCrossing
# - NonSignalTurn

# - EnterActorFlowV2
# - StaticCutIn
