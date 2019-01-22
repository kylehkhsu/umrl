#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python -m ipdb -c continue main_contextual.py \
    --algo ppo \
    --lr 3e-4 \
    --entropy-coef 0 \
    --value-loss-coef 0.5 \
    --ppo-epoch 10 \
    --num-mini-batch 32 \
    --gamma 0.99 \
    --tau 0.95 \
    --use-gae \
    --init-gain 2 \
    --policy-hidden-size 256 \
    --env-name HalfCheetahVel-v3 \
    --interface contextual \
    --rewarder unsupervised \
    --clusterer vae \
    --reward s_given_z \
    --conditional-coef 1 \
    --rewarder-fit-period 10 \
    --subsample-num 1024 \
    --subsample-strategy last-random \
    --subsample-last-per-fit 500 \
    --vae-beta 0.6 \
    --vae-lr 5e-4 \
    --vae-hidden-size 256 \
    --vae-latent-size 8 \
    --vae-layers 5 \
    --vae-plot \
    --vae-normalize-strategy adaptive \
    --vae-max-fit-epoch 1000 \
    --vae-batches 8 \
    --vae-marginal-samples 16 \
    --device cuda:0 \
    --seed 1 \
    --num-processes 10 \
    --trial-length 1 \
    --episode-length 100 \
    --trials-per-update 500 \
    --num-updates 500 \
    --save-period 10 \
    --vis-period 10 \
    --log_dir_root ./output/half-cheetah/exp_002 \
    --plot