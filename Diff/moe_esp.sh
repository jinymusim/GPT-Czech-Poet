#!/bin/bash
#PBS -q gpu -l select=1:ncpus=1:ngpus=1:mem=24gb:scratch_local=16gb
#PBS -l walltime=24:00:00
module add py-pip/py-pip-19.3-intel-19.0.4-hudzomi
module add intelcdk-17.1
module add cmake 
export TMPDIR=$SCRATCHDIR
singularity shell --nv /cvmfs/singularity.metacentrum.cz/NGC/PyTorch:22.10-py3.SIF
cd $SCRATCHDIR
pip install --target=$SCRATCHDIR -r /storage/brno2/home/chudobm/tf_shorts/Tensorflow-Shorts/Diff/requirements.txt
export PYTHONPATH="${PYTHONPATH}:${TMPDIR}"
cd /storage/brno2/home/chudobm/tf_shorts/Tensorflow-Shorts/Diff
python3 moediff_esp.py