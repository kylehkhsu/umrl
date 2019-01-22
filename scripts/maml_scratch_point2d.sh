#!/usr/bin/env bash
python -m ipdb -c continue main_maml.py \
    --log-dir-root ./output/point2d/exp_004 \
    --env-name Point2DWalls-corner-v0 \
    --rewarder supervised \
    --dense-coef 0.1 \
    --success-coef 10 \
    --num-adapt-val 500 \
    --fast-lr 0.1 \
    --fast-lr-val-after-one 0.1 \
    --fast-batch-size-val 20 \
    --num-processes-val 20 \
    --num-layers 2 \
    --hidden-size 64 \
    --bias-transformation-size 2 \
    --init-gain 1 \
    --num-batches 1 \
    --episode-length 50 \
    --gamma 0.99 \
    --tau 1.0 \
    --entropy-coef-val 0.001 \
    --device cuda:0 \
    --seed 1 \
    --plot \



