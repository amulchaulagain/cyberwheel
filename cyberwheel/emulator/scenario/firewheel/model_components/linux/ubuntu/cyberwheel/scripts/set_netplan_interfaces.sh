#!/bin/bash

#This script takes a list of MAC addresses and replaces them in
#/etc/netplan/firewheel.yaml with their device (interface) name

#A space separated string of MAC addresses to replace
SEARCH_MACS=$1

DEVS=()
MACS=()

get_interface_names () {
    DEVS=($(ip -o link show | awk -F': ' '{print $2}'))
}

get_macs () {
    for dev in ${DEVS[@]}
    do
        mac="$(ip address show $dev | grep ether | awk '{print $2}')"
        MACS+=("${mac}")
    done
}

find_device () {
    for (( i=0; i<${#MACS[@]}; i++ ));
    do
        if [ "${MACS[$i]}" == $1 ]
        then
            echo ${DEVS[$i]}
            return 0
        fi
    done

    echo ""
    return 1
}

get_interface_names
get_macs

for mac in $SEARCH_MACS
do
    dev=$(find_device $mac)
    echo "$mac=$dev"
    sed -i "s/$mac/$dev/g" /etc/netplan/firewheel.yaml
done

netplan apply
