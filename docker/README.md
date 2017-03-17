

# Run experiments in docker

[Docker](https://www.docker.com/) is a container solution that makes it easy to build and deploy 
software in a virtual environment. The examples in this folder use docker to easily deploy an experiment 
with all its dependencies, either on a local machine or on the cloud.

## Prerequisites

Install docker on your local machine by following the installation instructions for 
[Windows](https://docs.docker.com/docker-for-windows/install/), 
[Linux](https://docs.docker.com/engine/installation/), 
[MacOS](https://docs.docker.com/docker-for-mac/install/).

Prepare a docker machine on Azure, follow the local installation steps above, then run:
```
docker-machine create --driver azure --azure-size Standard_D12 --azure-subscription-id <subscription-id> <machine-name>
```
Replace `<subscription-id>` with your Azure subsciption id - you can find this on the Azure dashboard after 
logging on to https://portal.azure.com. The `<machine-name>` is arbitrary.

Additional `docker-machine` options are listed here: https://docs.docker.com/machine/drivers/azure/
Azure machine sizes are detailed on: https://docs.microsoft.com/en-us/azure/virtual-machines/virtual-machines-linux-sizes (we recommend to use at least size Standard_D12)

Configure docker to deploy to `<machine-name>`. Run:
```
docker-machine env <machine-name>
```
This will provide a script / instructions on how to prepare your environment to work with <machine-name>.

## Build the docker images

Build the required docker images:
```
cd docker
docker build malmo -t malmo:latest
docker build malmopy-cntk-cpu-py27 -t malmopy-cntk-cpu-py27:latest

```

Check to make sure that the images have been compiled:
```
docker images
```
You should see a list that includes the compiled images, e.g.,
```
REPOSITORY              TAG                          IMAGE ID            CREATED             SIZE
malmopy-cntk-cpu-py27   latest                       0161af81632d        29 minutes ago      5.62 GB
malmo                   latest                       1b67b8e2cfa8        41 minutes ago      1.04 GB
...
```

## Run the experiment

Run the challenge task with an example agent:
```
cd malmopy-ai-challenge
docker-compose up
```

The experiment is set up to start a tensorboard process alongside the experiment.
You can view it by pointing your browser to http://127.0.0.1:6006.

## Write your own

The provided docker files load malmopy and sample code directly from the
`malmo-challenge` git repository. To include your own code, create a file
called `Dockerfile` with the following content:

```
FROM malmopy-cntk-cpu-py27:latest

# add your own experiment code here
# ADD copies content from your local machine into the docker image
ADD ai_challenge/pig_chase /local/malmo-challenge/ai_challenge/pig_chase
```

Build this new image using:
```
docker build . -t my_malmo_experiment:latest
```

Point the `agents` service in `docker-compose.py` to the new image by replacing
`image: malmopy-cntk-cpu-py27:latest` with the name of the image you have just
built (e.g., `image:my_malmo_experiment:latest`). Also check if the working
directory or command need to be changed.

Then run the new experiment:
```
docker-compose up
```

## Cleaning up

If you are using a docker machine on Azure, make sure to shutdown and decomission 
the machine when your experiments have completed, to avoid incurring costs.

To shut a machine down:
```
docker-machine stop <machine-name>
```

To remove (decomission) a machine:
```
docker-machine rm <machine-name>
```

## Further reading

- [docker documentation](https://docs.docker.com/)
- [docker on Azure](https://docs.docker.com/machine/drivers/azure/)
- [docker compose](https://docs.docker.com/compose/overview/)
