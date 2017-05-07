#!/usr/bin/env bash
# runs experiment on machine NM, uses docker file ai_challenge dir
NM=${1}
docker-machine start ${NM}
docker-machine env ${NM}
eval $(docker-machine env ${NM})
cd ../..
docker build . -t my_exp
cd docker/my_dockerfiles
docker-compose down
docker-compose up