#!/usr/bin/env bash
cd ../..
docker build . -t my_exp
cd docker/my_dockerfiles
docker-compose up