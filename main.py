import numpy as np
import torch
import gym
import argparse
import os
import copy
from pathlib import Path
import time


from tqc import structures, DEVICE
from tqc.trainer import Trainer
from tqc.structures import Actor, Critic, RescaleAction
from tqc.functions import eval_policy, eval_policy_save_transition

import wandb


EPISODE_LENGTH = 1000


def main(args, results_dir, models_dir, prefix):
    # --- Init ---

    # remove TimeLimit
    env = gym.make(args.env).unwrapped
    eval_env = gym.make(args.env).unwrapped

    env = RescaleAction(env, -1., 1.)
    eval_env = RescaleAction(eval_env, -1., 1.)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    replay_buffer = structures.ReplayBuffer(state_dim, action_dim)
    actor = Actor(state_dim, action_dim).to(DEVICE)
    critic = Critic(state_dim, action_dim, args.n_quantiles, args.n_nets).to(DEVICE)
    critic_target = copy.deepcopy(critic)

    # top_quantiles_to_drop = args.top_quantiles_to_drop_per_net * args.n_nets
    top_quantiles_to_drop = args.top_quantiles_to_drop_per_net
    bottom_quantiles_to_drop = args.bottom_quantiles_to_drop_per_net * args.n_nets
    

    trainer = Trainer(args=args,
                      actor=actor,
                      critic=critic,
                      critic_target=critic_target,
                      top_quantiles_to_drop=top_quantiles_to_drop,
                      bottom_quantiles_to_drop=bottom_quantiles_to_drop,
                      move_mean_quantiles=args.move_mean_quantiles,
                      move_mean_from_origin=args.move_mean_from_origin,
                      discount=args.discount,
                      tau=args.tau,
                      target_entropy=-np.prod(env.action_space.shape).item(),
                      ens_type=args.ens_type,
                      )

    evaluations = []
    state, done = env.reset(), False
    episode_return = 0
    episode_timesteps = 0
    episode_num = 0

    st = time.time()
    total_st = time.time()

    file_name = f"{prefix}_{args.env}_{args.seed}"

    #####################
    if args.save_transition_mode:
        import glob
        # file = 'models/bottom0_top0_mean0_0/99_---'
        files = glob.glob(args.save_transition_dir + '/*_actor')
        files = [f.replace('_actor', "") for f in files]
        for file in files:
            print('----- file:', file, '-----')
            trans_file = file + "_transition"
            if os.path.exists(trans_file+'.npz'):
                print('skip')
                continue
            # file = models/bottom0_---/12'
            trainer.load(filename=file)
            print('model loaded.')
            avg_reward, trans = eval_policy_save_transition(actor, eval_env, EPISODE_LENGTH)
            print('evaluation done')
            np.savez(trans_file, state=trans[0], action=trans[1], reward=trans[2], done=trans[3])
            print('save done')
        return
    #####################

    actor.train()
    for t in range(int(args.max_timesteps)):
        action = actor.select_action(state)
        next_state, reward, done, _ = env.step(action)
        episode_timesteps += 1

        replay_buffer.add(state, action, next_state, reward, done)

        state = next_state
        episode_return += reward

        # Train agent after collecting sufficient data
        if t >= args.start_training_data:
            for _ in range(args.utd):
                trainer.train(replay_buffer, args.batch_size)

        if done or episode_timesteps >= EPISODE_LENGTH:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_return:.3f} Time: {time.time() - st:.1f}s Total: {time.time() - total_st:.1f}s")
            # Reset environment
            state, done = env.reset(), False

            episode_return = 0
            episode_timesteps = 0
            episode_num += 1

            st = time.time()

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            eval_st = time.time()
            evaluations.append(eval_policy(actor, eval_env, EPISODE_LENGTH))
            wandb.log({'evaluation_return': evaluations[-1]})
            np.save(results_dir / file_name, evaluations)
            st += time.time() - eval_st
        if args.save_model and (t + 1) % args.save_model_freq == 0:
            if not os.path.exists(models_dir / file_name):
                os.makedirs(models_dir / file_name)
            trainer.save(models_dir / file_name / str(t+1))
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Humanoid-v3")          # OpenAI gym environment name
    parser.add_argument("--eval_freq", default=None, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--save_model_freq", default=1e5, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=None, type=int)   # Max time steps to run environment
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--n_quantiles", default=25, type=int)
    parser.add_argument("--bottom_quantiles_to_drop_per_net", default=0, type=int)
    parser.add_argument("--top_quantiles_to_drop_per_net", default=2, type=int)
    parser.add_argument("--move_mean_quantiles", default=0, type=int)  # How many tiles we move the center one
    parser.add_argument("--move_mean_from_origin", action="store_true")
    parser.add_argument("--n_nets", default=5, type=int)
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)                 # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)                     # Target network update rate
    parser.add_argument("--log_dir", default='.')
    # parser.add_argument("--run_num", default=1)
    # parser.add_argument("--prefix", default='')
    parser.add_argument("--not_save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--save_transition_mode", action="store_true")
    parser.add_argument("--save_transition_dir", default='models/test')

    parser.add_argument("--start_training_data", default=256, type=int)
    parser.add_argument("--utd", default=1, type=int)

    parser.add_argument("--ens_type", default='tqc', choices=['tqc', 'ave', 'sample'])
    parser.add_argument("--qem", action="store_true")

    parser.add_argument("--prefix", default=None, type=str)


    args = parser.parse_args()

    if not args.max_timesteps:
        if args.env in ['Ant-v3', 'Walker2d-v3', 'HalfCheetah-v3']:
            args.max_timesteps = 5e6
        elif args.env in ['Humanoid-v3']:
            args.max_timesteps = 10e6
        elif args.env in ['Hopper-v3']:
            args.max_timesteps = 3e6
        else:
            raise ValueError
    if not args.eval_freq:
        args.eval_freq = 1e3
    if args.utd > 1:
        args.max_timesteps = int(args.max_timesteps / 10)
        args.eval_freq = int(args.eval_freq / 10)
        args.start_training_data = 5000
        

    if args.seed == -1:
        import random
        args.seed = random.randrange(10000)
        print('----- seed:', args.seed)


    args.save_model = not args.not_save_model

    log_dir = Path(args.log_dir)

    results_dir = log_dir / 'results'
    os.makedirs(results_dir, exist_ok=True)


    models_dir = log_dir / 'models'
    if args.save_model and not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # prefix = f'bottom{args.bottom_quantiles_to_drop_per_net}_top{args.top_quantiles_to_drop_per_net}_mean{args.move_mean_quantiles}_fromOrigin{args.move_mean_from_origin}_utd{args.utd}' #_run{args.run_num}'
    if args.prefix:
        prefix = args.prefix
    else:
        prefix = f'ensemble{args.n_nets}_top{args.top_quantiles_to_drop_per_net}_quantiles{args.n_quantiles}_utd{args.utd}_ENS{args.ens_type}_{"qemDim4" if args.qem else ""}_initData{args.start_training_data}'
        os.makedirs(os.path.join(results_dir, prefix), exist_ok=True)
        prefix = os.path.join(prefix, "")
    # prefix = 'test'
    # print('##################### test ###################')
    print(prefix)
    print(args)
    print(str(results_dir), prefix, prefix.strip(str(results_dir)))
    wandb_name = f'{prefix.replace(str(results_dir), "")}'
    print('wandb', wandb_name)
    wandb.init('tqc_average_ensemble', name=wandb_name)

    wandb.config.update(args)

    main(args, results_dir, models_dir, prefix)
    wandb.finish()

    # python main.py --env Walker2d-v2 --bottom_quantiles_to_drop_per_net 0 --top_quantiles_to_drop_per_net 2 --move_mean_quantiles 0 --seed 1
    # python main.py --env Walker2d-v2 --bottom_quantiles_to_drop_per_net 0 --top_quantiles_to_drop_per_net 2 --move_mean_quantiles 0 --save_transition_mode --save_transition_dir models/bot..
    # utd
    # python main.py --env Walker2d-v2 --bottom_quantiles_to_drop_per_net 0 --top_quantiles_to_drop_per_net 2 --move_mean_quantiles 0 --move_mean_from_origin --seed 0 --utd 20 --max_timesteps 30000000 --save_model_freq 10000 --start_training_data 5000

    # CUDA_VISIBLE_DEVICES="0" python main.py --env Ant-v3 --top_quantiles_to_drop_per_net 2 --n_nets 5 --ave_tiles --utd 1
    # CUDA_VISIBLE_DEVICES="0" python main.py --env Ant-v3 --top_quantiles_to_drop_per_net 2 --ens_type sample --qem