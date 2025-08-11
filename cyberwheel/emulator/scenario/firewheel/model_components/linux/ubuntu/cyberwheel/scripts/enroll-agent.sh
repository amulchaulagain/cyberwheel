#!/bin/bash


FLEET_SERVER_IP=$1
FLEET_SERVER_PORT=8220
AGENT_ENROLL_TOKEN=ckhQN1hKRUI5MFUzUDREcVJnNXI6bThlQVNBU0NUbEtscm5wbjhNTnJTZw==

sudo elastic-agent enroll \
    --url=$FLEET_SERVER_IP:$FLEET_SERVER_PORT \
    --enrollment-token=$AGENT_ENROLL_TOKEN \
    --insecure
