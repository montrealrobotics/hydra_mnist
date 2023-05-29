# MNIST with Hydra + Submitit + Lighting

This repository aims to show how to use Hydra, submitit and lighting to train models locally and in the cluster with a single python call. We assume familiarity with the Mila cluster and [CCBD](https://docs.alliancecan.ca/wiki/Getting_started).

## Overview 

[Hydra](https://hydra.cc/docs/intro/) allows to create dynamic hierarchic configurations. [Submitit](https://github.com/facebookincubator/submitit) is a library that allows to submit jobs to a cluster. [Lightning](https://www.pytorchlightning.ai/index.html) is a library that allows to train models in a standard way. However, if you want to replace lighting with Pytorch itself it is straightforward as long as you maintain the Hydra file structures. 

## Installation

### Locally

You just need to install the dependencies listed in `requirements.txt`. Note that right now it has a lot more than what it actually needs so feel free to prune it.

```bash
pip3 install -r requirements.txt
```

### Cluster

An entrypoint is required in order to launch the jobs using `python3`. This entrypoint depends on `Hydra` and therefore, we need to create a virtual environment with it. Install a [virtual environment in mila cluster](https://docs.mila.quebec/Userguide.html#pip-virtualenv) (or CCBD) with the required packages as follows:

```bash
# This was done for Mila but same applies for CCBD
module load python/3.8
python -m venv ./venv
source venv/bin/activate
# Install the required packages
pip install git+https://github.com/facebookincubator/submitit@escape_all#egg=submitit
pip install hydra-core==1.2.0
pip install hydra-colorlog==1.2.0
pip install git+https://github.com/MikeS96/hydra_submitit_launcher@main#egg=hydra-submitit-launcher
pip install hydra_optuna_sweeper==1.2.0
```

Once it is done, you are ready to run the code locally and in the cluster.

## Running the code

### Locally

To run the code locally, you do as if you were running a normal python script. For example, to run the mnist example you can do:

```bash
python3 train.py experiment=train_mnist epochs=5 model.net.hidden_dim=64
```

You can play with all the parameters defined in the configuration files and even sweep over those. This project has [comet-ml](https://www.comet.com/site/) configured to log stuff. If you want to attach it to the current run, simply try:

```bash
python3 train.py experiment=train_mnist  epochs=100 logger=comet logger.tag=dev
```

### Cluster

To run the code in the cluster, you need to activate the virtual environment created previously. Then, you need to attach the flag `--multirun` which indicated hydra multiple jobs will be executed and not a single one. Lastly, you define a configuration file, in this case `sweep_mnist` which will contain the grid of parameters you will sweep over. To launch it, simply do:

```bash
python3 train.py --multirun experiment=sweep_mnist launcher=mike_mila logger=comet logger.tag=cluster hydra.sweeper.n_trials=1
```

train.py