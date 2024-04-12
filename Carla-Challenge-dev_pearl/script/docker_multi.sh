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
export DEBUG_CHALLENGE=0
export REPETITIONS=1 # multiple evaluation runs
export RESUME=0
export HOST=192.168.193.102

ROUTE_SET=(0 2 4 8 10 12 14 16 20)
for i in ${!ROUTE_SET[@]}; do
    PORT=$((40100+3*i))
    TM_PORT=$((9000+3*i))
    kill -9 `lsof -i:$TM_PORT -t` > /dev/null 2>&1

    gnome-terminal --geometry=80x24 -- /bin/bash -c "\
    source activate carla-rl;
    cd ./src/decision;
    python3 ./custom_evaluator.py \
    --routes=${LEADERBOARD_ROOT}/data/routes_training.xml \
    --repetitions=${REPETITIONS} \
    --track=${CHALLENGE_TRACK_CODENAME} \
    --agent=${WORKSPACE}/src/decision/agent/agent_single.py \
    --agent-config=${WORKSPACE}/src/decision/configs/config_single.yaml \
    --debug=${DEBUG_CHALLENGE} \
    --resume=${RESUME} \
    --routes-subset=${ROUTE_SET[i]} \
    --host=${HOST} \
    --port=${PORT} \
    --traffic-manager-port=${TM_PORT};
    exec bash"
done




