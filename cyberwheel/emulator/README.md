# Emulator

## Overview

For now Firewheel must be started up manually before running Agent evaluation. Follow the instructions in Setup and Starting Firewheel Experiment before starting the agent evaluation.

## Setup

**Uploading a network config file**:

1. ssh into `firewheel-dev01` server
2. pull the cyberwheel repository if not done already
3. `cp cyberwheel/cyberwheel/resources/configs/network/<file>.yaml` `/opt/firewheel/model_components/cyberwheel/topology/configs/` 
    - this may require `sudo`

This only needs to be done once. `example_config.yaml` and `integration_config.yaml` are already copied over.

## Starting Firewheel Experiment

Below are the steps to start an experiment:

1. ssh into `firewheel-dev01` server
2. `sudo su - fw`
    - use your ucams password
3. `cd /opt/firewheel/model_components/cyberwheel/topology/`
4. `./run.sh <config-name>.yaml` (ensure the network topology file is in the `configs` folder, see Setup above)
5. use `firewheel vm mix` to check when virtural machines are finsihed configuration. The _VM Resource State_ should all say "configured".

Once all machines are configured. You must run the **EmulatorController** class `init_hosts()` method to connect hosts to Elastic's Fleet Server. (This will be automatic eventually).

6. switch back to your own username
7. cd in the test directory in the cyberwheel repostity `cd cyberwheel/cyberwheel/tests/`
8. edit `test_emulator_control.py` to ensure `NETWORK_CONFIG` is the correct config file.
9. run `poetry run python -m unittest -v test_emulator_control.TestEmulatorSetup.test_init_hosts`

It should show all hosts (excluding decoy's) being connected to Fleet. Now you start the agent evaluation script.

## Resetting Firewheel

To stop experiment run `firewheel restart`. It's best to do this when you're done testing since Elastic is running in the SIEM virtual machine and will continue to log. The memory will eventually run out.

## Portfording MiniMega

If you want to visually see the virtual machines and remote into them, you can use MiniMega through the browser. You'll need need to ssh into firewheel-dev01 with port forwarding first:

1. `ssh -L localhost:9011:localhost:9001 <firewheel-dev01>`
2. open up a browser to `http://localhost:9001`

## SSH into a virtual machine in Firewheel

After starting an experiment and machines are configured, you can SSH into any machine. Run the following command and use the password `ubuntu`:

`firewheel ssh ubuntu@<hostname>`  
