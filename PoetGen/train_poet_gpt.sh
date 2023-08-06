#!/bin/bash
#PBS -q gpu -l select=1:ncpus=1:ngpus=1:mem=24gb:scratch_local=16gb
#PBS -l walltime=24:00:00
module add py-pip/21.3.1-gcc-10.2.1-mjt74tn
export TMPDIR=$SCRATCHDIR
singularity shell --nv /cvmfs/singularity.metacentrum.cz/NGC/PyTorch:23.05-py3.SIF
cd $SCRATCHDIR
pip install --target=$SCRATCHDIR torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install --target=$SCRATCHDIR -r /storage/brno2/home/chudobm/tf_shorts/Tensorflow-Shorts/PoetGen/requirements.txt
export PYTHONPATH="${PYTHONPATH}:${TMPDIR}"
cd /storage/brno2/home/chudobm/tf_shorts/Tensorflow-Shorts/PoetGen
python3 lm_finetune_torch_trainer_api.py