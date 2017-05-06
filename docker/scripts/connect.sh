#!/usr/bin/env bash
# connects to machine NM
NM=${1}
docker-machine start ${NM}
docker-machine env ${NM}
eval $(docker-machine env ${NM})
docker exec -it $(docker ps -l -q) bash