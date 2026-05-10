#!/usr/bin/env bash
set -euo pipefail

nvcc -std=c++20 -O3 --restrict main.cu -o mrlbm
./mrlbm
