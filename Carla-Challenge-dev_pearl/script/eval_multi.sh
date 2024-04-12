#!/bin/bash

export WORKSPACE=${PWD}
export CARLA_ROOT=${WORKSPACE}/../CARLA
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export GPU_INDEX=0
export SERVER_PORT=2000
export TM_PORT=2500

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

CARLA_PORTS=()
TM_PORTS=()

tmp=$((20*${GPU_INDEX}))
for i in {0..2}; do
    CARLA_PORTS+=($((${SERVER_PORT}+${tmp}+3*${i})))
    TM_PORTS+=($((${TM_PORT}+${tmp}+3*${i})))
done
unset tmp

# kill the process of previous simulation to avoid blind error
# Note: the following command will kill all process using specific GPU
kill -9 `nvidia-smi -i ${GPU_INDEX} --query-compute-apps=pid --format=csv,noheader | tr '\n' ' '` > /dev/null 2>&1
for port in ${TM_PORTS[@]}; do
    kill -9 `lsof -i:$port -t` > /dev/null 2>&1
done
export CUDA_VISIBLE_DEVICES=${GPU_INDEX}

# load multiple CARLA servers
carla_command=""
for port in ${CARLA_PORTS[@]}; do
    # carla_command+="${CARLA_SERVER} -carla-rpc-port=$port -windowed -ResX=400 -ResY=300 & "
    carla_command+="${CARLA_SERVER} -carla-rpc-port=$port -ini:[/Script/Engine.RendererSettings]:r.GraphicsAdapter=${GPU_INDEX} -RenderOffScreen & "
done
gnome-terminal -- bash -c "${carla_command%??}"

# wandb default setting file: ~/.netrc
# wait for CARLA servers
sleep 30
echo "> In eval_multi.sh: CARLA loading done."

export CHALLENGE_TRACK_CODENAME=SENSORS
export DEBUG_CHALLENGE=0
export REPETITIONS=1 # multiple evaluation runs
export RESUME=0

cd ./src/decision
python3 ./train.py \
--routes=${LEADERBOARD_ROOT}/data/routes_training.xml \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--agent=${WORKSPACE}/src/decision/agent/agent_multi.py \
--debug=${DEBUG_CHALLENGE} \
--resume=${RESUME} \
--port=${SERVER_PORT} \
--traffic-manager-port=${TM_PORT} \
--gpu-index=${GPU_INDEX} \

