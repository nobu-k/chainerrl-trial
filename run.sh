#!/bin/bash

set -e
apt-get update
apt-get install -y xvfb libav-tools
echo "starting ChainerRL..."
xvfb-run -s "-screen 0 1400x900x24" python3 main.py
