# Instructions to run Game of Tag example

## Build manually and run locally
Currently, do not install `smarts` directly from pypi. Instead, install `smarts` from the `smarts-got-*` branch of `huaweinoah/smarts` repository. 

The steps to install and run smarts-got is as follows.

```bash
# Install Smarts and build scenarios
$ cd /path/to/SMARTS/
$ git clone https://github.com/huawei-noah/SMARTS.git 
$ git checkout smarts-got-*
# Follow prompts for setting up sumo and SUMO_HOME environment variable
$ ./install_deps.sh
# Verify sumo is >= 1.5.0
$ sumo
$ python3.7 -m venv .venv
$ source ./.venv/bin/activate
$ pip install --upgrade pip
$ pip install -e .[dev,camera-obs]
$ pip install -r ./examples/gameOfTag/requirements.txt
$ scl scenario build-all ./scenarios

# Run the code 
$ python3.7 ./examples/gameOfTag/train.py
```

## To visualise in Envision
1. set `headless: False` in `./examples/gameOfTag/got.yaml` file
2. Run `scl` in a separate terminal
    ```bash
    $ scl envision start -s ./scenarios
    ```
3. Navigate to localhost:8081

## Build and run in docker container
```bash
# In host terminal
$ cd /path/to/SMARTS

$ export GOTVERSION=v0.4.19

$ docker build --network=host -f ./utils/docker/Dockerfile.tensorflow -t adaickalavan/smarts:v0.4.19-tensorflow .

$ docker build --network=host -f ./utils/docker/Dockerfile.tensorflow -t test .

$ docker run --rm -it --gpus=all --network=host --volume=/home/kyber/workspaces/SMARTS/:/src/ adaickalavan/smarts:v0.4.19-tensorflow

$ docker run --rm -it --gpus=all --network=host --volume=/home/kyber/workspaces/SMARTS/examples/gameOfTag/logs:/src/examples/gameOfTag/logs --volume=/home/kyber/workspaces/SMARTS/examples/gameOfTag/models:/src/examples/gameOfTag/models adaickalavan/smarts:v0.4.19-tensorflow

$ docker run --rm -it --gpus=all --network=host --volume=/home/adai/workspaces/SMARTS/:/src/ adaickalavan/smarts:v0.4.19-tensorflow

$ docker run --rm -it --gpus=all --network=host --volume=/data:/data2  got:2021_10_12_10_53 

# In interactive docker container bash 
$ cd /src
# To train
$ PYTHONHASHSEED=0 python3.7 ./examples/gameOfTag/train.py
```

## Tensorboard
```
$ tensorboard --logdir=/home/adai/workspaces/SMARTS/examples/gameOfTag/logs
$ tensorboard --logdir=/home/adai/workspaces/training/got_2021_09_27_03_36/logs
```

## Training
1. got_2021_10_12_16_17
- observation: two rgb images 

1. got_2021_10_12_10_53
- observation: two rgb images
- subprocvecenv