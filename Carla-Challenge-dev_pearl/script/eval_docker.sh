#!/bin/bash
source activate reqtest;

export WORKSPACE=${PWD}
export CARLA_ROOT=${WORKSPACE}/../CARLA
export HOST_IP=192.168.193.102
export SERVER_PORT=40100
export TM_PORT=8500
export CONFIG_PATH=

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

kill -9 `lsof -i:${TM_PORT} -t` > /dev/null 2>&1

export CHALLENGE_TRACK_CODENAME=SENSORS
export DEBUG_CHALLENGE=0
export REPETITIONS=1 # multiple evaluation runs
export RESUME=0

cd ./src/decision
python3 ./custom_evaluator.py \
--routes=${LEADERBOARD_ROOT}/data/routes_training.xml \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--agent=${WORKSPACE}/src/decision/agent_dsac.py \
--agent-config=${CONFIG_PATH} \
--debug=${DEBUG_CHALLENGE} \
--resume=${RESUME} \
--host=${HOST_IP} \
--port=${SERVER_PORT} \
--timeout=100 \
--traffic-manager-port=${TM_PORT} \
--routes-subset="0,5,6,7"

