#!/usr/bin/env bash
# creates a instance in the cloud and runs docker container, uses docker file ai_challenge dir
# sample usage: bash create_run_exp.sh m1 1 DQN rec_value_based_exp
NM=${1}
REPS=${2}
MOD=${3}
EXP=${4}
SUB="" # copy sub ID here
docker-machine create --driver azure --azure-size Standard_F8 --azure-subscription-id ${SUB} ${NM}
docker-machine env ${NM}
eval $(docker-machine env ${NM})
cd ..
docker build malmo -t malmo:latest
docker build malmopy-cntk-cpu-py27 -t malmopy-cntk-cpu-py27:latest
cd ..
docker build . -t my_exp
cd docker/my_dockerfiles
python run_docker_compose.py -e ${EXP} -m ${MOD} r${REPS}
