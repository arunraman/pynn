#!/bin/bash
# export environment variables


pynndir=/ws/

export PYTHONPATH=$pynndir
export OMP_NUM_THREADS=4

pythonCMD="python -u -W ignore"

mkdir -p model

CUDA_VISIBLE_DEVICES=0 $pythonCMD export_seq2seq.py \
                                  --n-classes 4003 --d-input 40 --d-model 1024 --n-enc 6 \
                                  --use-cnn --freq-kn 3 --freq-std 2 --mean-sub \
                                  --n-dec 2 --n-head 1 --export-onnx --batch-size 32 --test-onnx \
                                  --test-pytorch
