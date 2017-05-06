#!/usr/bin/env bash
# copies results from machine NM to folder RES_NM
NM=${1}
RES_NM=${2}
docker-machine start ${NM}
docker-machine env ${NM}
eval $(docker-machine env ${NM})
docker cp $(docker ps -l -q):/root/malmo-challenge/ai_challenge/results ${RES_NM}