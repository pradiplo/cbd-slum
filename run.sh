#!/bin/bash
#--- parameter define
#SBATCH -J cbd
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -c 32

#--- Job Script
python3 read.py