#!/bin/bash

set -e
apt-get install -y xvfb libav-tools
xvfb-run -s "-screen 0 1400x900x24" python main.py
