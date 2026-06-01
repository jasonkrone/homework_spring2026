#!/bin/bash


function 6dot0() {

    # i'm okay to do ~300 runs

    # --discount [1.0, 0.99, 0.9]
    # try w/ normalize advantage off
    # Whether to use return-to-go (default off)
    # Whether to use GAE, and if we do, what value of λ to use (default off)
    ########### defer till later above this line

    # Deploying registers hw2_modal_remote as a persistent function. When the local entrypoint then calls spawn(), those tasks are submitted to the
    # deployed app and run independently — your local process exits and the jobs keep going.
    #uv run modal deploy src/scripts/modal_run.py
    #uv run modal run src/scripts/modal_run.py
    # uv run modal app stop hw2-pg
    uv run modal run --detach src/scripts/modal_run.py

}


function 5dot0() {
    for lambda in 0 0.95 0.98 0.99 1; do
        uv run src/scripts/run.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 -n 200 -b 2000 -eb 2000 -l 3 -s 128 -lr 0.001 --use_reward_to_go --use_baseline --gae_lambda $lambda --exp_name discounted_lunar_lander_lambda$lambda --no_gpu
    done
}


function 4dot2() {
    #uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg --discount 0.95 -lr 0.01 --exp_name cheetah
    #uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg --discount 0.95 -lr 0.01 --use_baseline -blr 0.01 -bgs 5 --exp_name cheetah_baseline
    #uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg --discount 0.95 -lr 0.01 --use_baseline -blr 0.001 -bgs 5 --exp_name cheetah_baseline_low_blr
    #uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg --discount 0.95 -lr 0.01 --use_baseline -blr 0.01 -bgs 1 --exp_name cheetah_baseline_low_bgs
    #uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg --discount 0.95 -lr 0.01 --use_baseline -blr 0.001 -bgs 1 --exp_name cheetah_baseline_low_blr_bgs
    #uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -na -eb 3000 -rtg --discount 0.95 -lr 0.01 --use_baseline -blr 0.01 -bgs 5 --exp_name cheetah_baseline_na
    uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg --discount 0.95 -lr 0.01 --use_baseline -blr 0.01 -bgs 5 --video_log_freq 10 --exp_name cheetah_baseline_videos
}

function 3dot2() {
    # n: iters
    # b: batch size
    # rtg: use reward to go
    # na: normalize advantage
    # 
    uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name cartpole
    uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 -rtg --exp_name cartpole_rtg
    uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 -na --exp_name cartpole_na
    #uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 -rtg -na --exp_name cartpole_rtg_na
    #uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 4000 --exp_name cartpole_lb
    #uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 4000 -rtg --exp_name cartpole_lb_rtg
    #uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 4000 -na --exp_name cartpole_lb_na
    #uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 4000 -rtg -na --exp_name cartpole_lb_rtg_na
}