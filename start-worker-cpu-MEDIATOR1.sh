#!/bin/bash

SYSTEM_PATH=`dirname "$0"`

export LD_LIBRARY_PATH=$SYSTEM_PATH/lib
export PYTHONPATH=$SYSTEM_PATH/pylib

#pythonCMD="python -u -W ignore"
pythonCMD="/home/tnguyen/asr/anaconda3/envs/pytorch1.7/bin/python -u -W ignore"
log=logs/asr-en-mediator1.worker.log

CUDA_VISIBLE_DEVICES=$2 OMP_NUM_THREADS=8 $pythonCMD worker.py \
	--server "localhost" \
	--port 60020 \
	--name "asr-en" \
	--fingerprint "en-EU-memory$1" \
	--outfingerprint "en-EU" \
        --inputType "audio" \
        --outputType "text" \
        --dict "model/bpe4k.dic" \
        --model "model/s2s-lstm.dic" \
        --punct "model/punct.dic" \
        --device 'cuda' --beam-size 4 --new-words "http://ufallab.ms.mff.cuni.cz/~kumar/elitr/live-adaptation/memory-$1.txt" 
