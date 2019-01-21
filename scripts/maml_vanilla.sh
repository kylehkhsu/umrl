#!/usr/bin/env bash
python -m ipdb -c continue main_maml_vanilla.py \
    --env-name HalfCheetahDir-v1 \
    --num-workers 8 \
    --fast-lr 0.1 \
    --max-kl 0.01 \
    --fast-batch-size 20 \
    --meta-batch-size 40 \
    --num-layers 2 \
    --hidden-size 256 \
    --num-batches 1000 \
    --gamma 0.99 \
    --tau 1.0 \
    --cg-damping 1e-5 \
    --ls-max-steps 15 \
    --device cuda:0 \
    --seed 1