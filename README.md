Experiments can be run locally, or with the use of docker. 

- To run experiments with docker, go to docker folder and follow instructions
in README.md to build required images. Then build the Dockerfile in this directory and run
```docker-compose up``` in docker/my_dockerfiles

- To run experiments locally, make sure that you have project Malmo installed.
Then install from setup.py (e.g. run ```pip install -e .```). You should
be able to run ai_challenge/main.py after that