# Emulator

## Overview

Cyberwheel uses [Firewheel](https://github.com/sandialabs/firewheel) for emulation and evaluating agents. Firewheel provides a high-fidelity testing environment to perform repeatable experiments and scale to large networks.

We reccomended after installing Firewheel, to go through their [quickstart](https://sandialabs.github.io/firewheel/quickstart.html) and [tutorials](https://sandialabs.github.io/firewheel/tutorials/index.html) to get familiar with starting an emulated network environment, what they refer to as an _experiment_.

## Hardware Pre-requisites

- **16GB RAM minimum** (reccomended)
- **Linux OS**: Ubuntu, CentOS, RHEL

Firewheel uses QEMU/KVM to host virtual machines (VM) and the default memory allocated for each VM is 2GB. 
The memory is adjustable in the `model_component_object.py` definitions located in the `/opt/firewheel/model_components` folder.

For Cyberwheel, we defined 2 custom components:

- **Ubuntu2204DesktopHostCyberwheel** (2GB)- used for all hosts (e.g. users, servers, workstations), except the SIEM host. It uses an Ubuntu 22.04.4 image as its base image. The image is pre-installed with software to perform Red Actions such as `nmap` for port scans. 
- **Ubuntu1604DesktopSiemcyberwheel** (4GB) - used for the SIEM host. It uses an Ubuntu 16.04.4 image as its base image. The image is pre-installed with ElasticSearch, Kibana and Fleet.

## Software Pre-requisites

- [Firewheel](https://github.com/sandialabs/firewheel): follow the [installation guide](https://sandialabs.github.io/firewheel/install/install.html) to install Firewheel and its dependencies. Currently Firewheel has been tested on Ubuntu, CentOS and RHEL. 
- [sshpass](https://www.cyberciti.biz/faq/noninteractive-shell-script-ssh-password-provider/): passes the host password when SSH'ing into a host, which skips having to enter the password. 
This is for the _Action Controller_ to perform actions on hosts within Firewheel.


## Before Starting

Firewheel has its own command line interface (CLI) and references model componnets defined in `/opt/firewheel/model_components` when starting experiment. 
We must copy the **cyberwheel model component** (scenario converter) and **images** from the repository into the `model_components` folder.

**Copy the model component**:

```
mkdir /opt/firewheel/model_components/cyberwheel
cd /path-to-repo/cyberwheel
cp -r cyberwheel/emulator/scenario/firewheel/toplogy /opt/firewheel/model_components/cyberhweel
```

**Copy the images**:

```
cd /path-to-repo/cyberwheel
cp -r cyberwheel/emulator/scenario/firewheel/model_components/linux/ubuntu/cyberwheel /opt/firewheel/model_components/linux/ubuntu
```

## Starting Firewheel

Before evaluating a trained agent, the emulation envinronment in Firewheel must be running:

```
sudo su - fw
cd /opt/firewheel/model_components/cyberwheel/topology
./run.sh path-to-repo/cyberwheel/data/configs/network/<network-config>.yaml 

# example 
./run.sh /home/cyberwheel/cyberwheel/data/configs/network/emulator_example_config.yaml)
```

Check the status of virtural machines are finsihed configuration 
The _VM Resource State_ should all say _configured_.

```
firewheel vm mix  # check status of VMs
ctrl+c            # exit status check
```

## Evaluating the Agent

Once all machines are configured. Open a new terminal (no need to switch to `fw` user):

```
cd path-to-repo/cyberwheel
poetry install   # if you have not already installed dependencies
poetry shell     # enter virtual environment
python cyberwheel/emulator/evaluate_cyberwheel_emu.py --eval-config <evaluation-config>.yaml   # run evaluation

# example 
python cyberwheel/emulator/evaluate_cyberwheel_emu.py --eval-config emulator_test_integration.yaml
```

## Resetting Firewheel

To gracefully shutdown the VMs and reset the emulation environment run `firewheel restart`. 
If you leave the the network working the SIEM host with Elastic will continue to collect logs from the other hosts. 
The memory will eventually run out.

## Architecture Overiew

### Important Folders and Files

- `actions/`: contains emulator actions for both red and blue agents.
- `configs/emulator_config.yaml`:  contains configs such as host username and password.
- `controler/emulator_contoler.yaml`: class for the simulator to interface with the emulator.
- `detectors/emulator_dectector.py`: class that defines the emulator detector.
- `scenario/`: contains the network topology (i.e. scenario) converter and custom KVM images used by firewheel.

### Scenario Converter

(TODO)

### Action Controller

(TODO)

### Detector and Observation Controller

(TODO)
