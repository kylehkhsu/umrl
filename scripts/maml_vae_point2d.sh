#!/usr/bin/env bash
python -m ipdb -c continue main_maml.py \
    --log-dir-root ./output/point2d/exp_004 \
    --env-name Point2DWalls-corner-v0 \
    --rewarder unsupervised \
    --clusterer vae \
    --reward s_given_z \
    --conditional-coef 1 \
    --dense-coef 0.1 \
    --success-coef 10 \
    --num-adapt-val 4 \
    --fast-lr-val-after-one 0.05 \
    --fast-batch-size-val 20 \
    --fast-lr 0.1 \
    --num-layers 2 \
    --hidden-size 64 \
    --bias-transformation-size 2 \
    --init-gain 1 \
    --num-batches 500 \
    --subsample-num 1024 \
    --subsample-strategy last-random \
    --vae-beta 0.5 \
    --vae-lr 5e-4 \
    --vae-hidden-size 256 \
    --vae-latent-size 4 \
    --vae-layers 5 \
    --vae-plot \
    --vae-normalize-strategy adaptive \
    --vae-max-fit-epoch 2000 \
    --vae-batches 1 \
    --vae-marginal-samples 16 \
    --num-processes 20 \
    --episode-length 50 \
    --rewarder-fit-period 20 \
    --save-period 5 \
    --vis-period 5 \
    --val-period 10 \
    --fast-batch-size 20 \
    --meta-batch-size 40 \
    --gamma 0.99 \
    --tau 1.0 \
    --entropy-coef 0.001 \
    --entropy-coef-val 0.001 \
    --device cuda:0 \
    --seed 1 \
    --plot \



