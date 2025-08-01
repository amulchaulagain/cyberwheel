# Emulator

## Overview

<figure align="center">
<img 
  alt="Cyberwheel Emulation Environment Diagram"
  src="images/emulation-environment.png" 
  width="400"
>
</figure>

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
- `controler/emulator_contoler.py`: class for the simulator to interface with the emulator.
- `detectors/emulator_dectector.py`: class that defines the emulator detector.
- `scenario/`: contains the network topology (i.e. scenario) converter and custom KVM images used by firewheel.

### Scenario Converter

The `scenario/firewheel/toplogy/plugin.py` file converts a network config file into a firewheel experiment that consists of subnets and hosts.
There are two model components (MCs) used to create the hosts: 1) _Ubuntu2204DesktopHostCyberwheel_ and 2) _Ubuntu1604DesktopSiemCyberwheel_.
The MCs use custom Kernal-based Virtual Machine (KVM) images that have software installed:

- Ubuntu2204DesktopHostCyberwheel
    - [nmap](https://nmap.org/) - for _port scan_ action
    - [ccencrypt](https://manpages.debian.org/jessie/ccrypt/ccencrypt.1.en.html) - for _Data Encrypted for Impact_ action
    - [sysmon](https://github.com/microsoft/SysmonForLinux) - system logger
    - [Elastic Agent](https://www.elastic.co/docs/reference/fleet) - sends sysmon logs to fleet server

- Ubuntu1604DesktopSiemCyberwheel
    - [Elasticsearch](https://www.elastic.co/elasticsearch) - ingest logs and acts as a SIEM.
    - [Kibana](https://www.elastic.co/kibana) - visual analytics tools for elasticsearch and also used to setup Fleet.
    - [Fleet Server](https://www.elastic.co/docs/reference/fleet/) - server for Elastic Agents to connect to and collect sysmon logs.


To learn more about creating Firewheel experiments and model components, we recommend following the official Firewheel [tutorials](https://firewheel-docs.ornl.gov/tutorials/index.html). 

To learn more about creating custom images, follow Firewheel guides [here](https://firewheel-docs.ornl.gov/tutorials/image.html).

### Action Controller

Blue and Red agent actions are stored in `actions/` and executed through the class in  `control/emulator_contoler.py`.
All attacks are written to be executed on linux machines, since all host are using an Ubuntu image. 
The included actions are listed below. We encourge users to create and add more actions.

- Blue Agent Actions
    - `emulate_deploy_decoy_host.py` - deploys a decoy by turning on the network interface* 
    - `emulate_remove_decoy_host.py` - removes a decoy by turning off the network interface

- Red Agent Actions
    - `emulate_ping_sweep.py` - ping sweep for a specified subnet.
    - `emulate_port_scan.py` - port scan using nmap on a specified host.
    - `emulate_sudo_and_sudo_caching.py` - privilege escalation described in [T1548.003](https://www.atomicredteam.io/atomic-red-team/atomics/T1548.003)
    - `emulate_data_encrypted_for_impact.py` - encryption attack described in [T1486](https://www.atomicredteam.io/atomic-red-team/atomics/T1486)

*_all host (VMs) are created before an experiment starts, including decoys. Firewheel does not allow a VM to be created dynamically after starting an experiment.
To emulate deploying and removing a decoy, the Blue Agent enables and disables the ens2 interface._

Actions are executed through ssh. Firewheel provides a CLI command `firewheel ssh` to ssh into a host within a running experiment.
Before an action is executed, _firewheel ssh_ is first used to ssh into the target host. This step is defined in `actions/{red, blue}_action_base.py`.
To avoid having to manually enter the password "_ubuntu_" during ssh, `sshpass` passes the password as part of the action command. 
That said, remember to **install sshpass** on the same machine as Firewheel

### Detector and Observation Controller

(TODO)
