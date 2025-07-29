#!/bin/bash

sudo elastic-agent enroll \
  --fleet-server-es=http://localhost:9200 \
  --fleet-server-service-token=AAEAAWVsYXN0aWMvZmxlZXQtc2VydmVyL3Rva2VuLTE3MjM4NDEyMTM4MjQ6YnFYLTA4M2NRVk93aGFPZW5ERDdpZw \
  --fleet-server-policy=fleet-server-policy \
  --fleet-server-port=8220
