#!/usr/bin/env bash
python -m ipdb -c continue main_maml.py \
    --log-dir-root ./output/point2d/exp_003 \
    --env-name Point2DWalls-corner-v0 \
    --rewarder supervised \
    --dense-coef 0.1 \
    --success-coef 10 \
    --num-adapt-val 4 \
    --fast-lr-val-after-one 0.05 \
    --fast-batch-size-val 20 \
    --num-processes 20 \
    --fast-lr 0.1 \
    --num-layers 2 \
    --hidden-size 64 \
    --bias-transformation-size 2 \
    --num-batches 500 \
    --save-period 5 \
    --val-period 10 \
    --vis-period 5 \
    --plot \
    --episode-length 50 \
    --fast-batch-size 20 \
    --meta-batch-size 40 \
    --gamma 0.99 \
    --tau 1.0 \
    --entropy-coef 0.001 \
    --entropy-coef-val 0.001 \
    --cg-damping 1e-5 \
    --max-kl 0.01 \
    --ls-max-steps 15 \
    --device cuda:0 \
    --seed 1 \


