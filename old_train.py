import gym
import torch
import json
import os
import yaml
from tqdm import trange

import maml_rl.envs
from maml_rl.metalearners import MAMLTRPO
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.samplers import MultiTaskSampler
from maml_rl.utils.helpers import get_policy_for_env, get_input_size
from maml_rl.utils.reinforcement_learning import get_returns


def main(args):
    #with open(args.config, 'r') as f:
    #    config = yaml.load(f, Loader=yaml.FullLoader)
    '''
    if args.output_folder is not None:
        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)
        policy_filename = os.path.join(args.output_folder, 'policy.th')
        config_filename = os.path.join(args.output_folder, 'config.json')

        with open(config_filename, 'w') as f:
            config.update(vars(args))
            json.dump(config, f, indent=2)
    '''
    '''
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    '''
    env = gym.make(args.env_name,**{"num_bits":7})#**config.get('env-kwargs', {}))
    env.close()

    # Policy
    policy = get_policy_for_env(env,
                                hidden_sizes=args.hidden_size,
                                nonlinearity=args.nonlinearity)
    policy.share_memory()

    # Baseline
    baseline = LinearFeatureBaseline(get_input_size(env))

    # Sampler
    sampler = MultiTaskSampler(config['env-name'],
                               env_kwargs=config.get('env-kwargs', {}),
                               batch_size=config['fast-batch-size'],
                               policy=policy,
                               baseline=baseline,
                               env=env,
                               seed=args.seed,
                               num_workers=args.num_workers)

    metalearner = MAMLTRPO(policy,
                           fast_lr=config['fast-lr'],
                           first_order=config['first-order'],
                           device=args.device)

    num_iterations = 0
    for batch in trange(config['num-batches']):
        tasks = sampler.sample_tasks(num_tasks=config['meta-batch-size'])
        futures = sampler.sample_async(tasks,
                                       num_steps=config['num-steps'],
                                       fast_lr=config['fast-lr'],
                                       gamma=config['gamma'],
                                       gae_lambda=config['gae-lambda'],
                                       device=args.device)
        logs = metalearner.step(*futures,
                                max_kl=config['max-kl'],
                                cg_iters=config['cg-iters'],
                                cg_damping=config['cg-damping'],
                                ls_max_steps=config['ls-max-steps'],
                                ls_backtrack_ratio=config['ls-backtrack-ratio'])

        train_episodes, valid_episodes = sampler.sample_wait(futures)
        num_iterations += sum(sum(episode.lengths) for episode in train_episodes[0])
        num_iterations += sum(sum(episode.lengths) for episode in valid_episodes)
        logs.update(tasks=tasks,
                    num_iterations=num_iterations,
                    train_returns=get_returns(train_episodes[0]),
                    valid_returns=get_returns(valid_episodes))

        # Save policy
        if args.output_folder is not None:
            with open(policy_filename, 'wb') as f:
                torch.save(policy.state_dict(), f)


if __name__ == '__main__':
    import argparse
    import multiprocessing as mp
    import os    

    parser = argparse.ArgumentParser(description='Reinforcement learning with '
        'Model-Agnostic Meta-Learning (MAML)')

    # General
    parser.add_argument('--env-name', type=str, default='BitFlipEnv-v0',
        help='name of the environment')
    parser.add_argument('--gamma', type=float, default=0.9,
        help='value of the discount factor gamma')
    parser.add_argument('--tau', type=float, default=0.99,
        help='value of the discount factor for GAE')
    parser.add_argument('--first-order', action='store_true',
        help='use the first-order approximation of MAML')

    # Policy network (relu activation function)
    parser.add_argument('--hidden-size', type=int, default=100,
        help='number of hidden units per layer')
    parser.add_argument('--num-layers', type=int, default=2,
        help='number of hidden layers')
    parser.add_argument('--nonlinearity', type=str, default='relu',
        help='non linearity spec')

    # Task-specific
    parser.add_argument('--fast-batch-size', type=int, default=3, # 17
        help='batch size for each individual task')
    parser.add_argument('--fast-lr', type=float, default=0.1,
        help='learning rate for the 1-step gradient update of MAML')

    # Optimization
    parser.add_argument('--num-batches', type=int, default=200,
        help='number of batches')
    parser.add_argument('--meta-batch-size', type=int, default=1, #22
        help='number of tasks per batch')
    parser.add_argument('--max-kl', type=float, default=1e-2,
        help='maximum value for the KL constraint in TRPO')
    parser.add_argument('--cg-iters', type=int, default=10,
        help='number of iterations of conjugate gradient')
    parser.add_argument('--cg-damping', type=float, default=1e-5,
        help='damping in conjugate gradient')
    parser.add_argument('--ls-max-steps', type=int, default=15,
        help='maximum number of iterations for line search')
    parser.add_argument('--ls-backtrack-ratio', type=float, default=0.5,
        help='maximum number of iterations for line search')

    # Miscellaneous
    parser.add_argument('--output-folder', type=str, default='maml-Bitflip-dir',
        help='name of the output folder')
    parser.add_argument('--output-traj-folder', type=str, default='Bitflip-traj-dir',
        help='name of the output trajectory folder')
    parser.add_argument('--save_every', type=int, default=20,     
                        help='save frequency')
    parser.add_argument('--num-workers', type=int, default=8,
        help='number of workers for trajectories sampling')
    parser.add_argument('--device', type=str, default='cuda',
        help='set the device (cpu or cuda)')
    parser.add_argument('--resume_training', type=bool, default=False,
        help='if want to resume training from a saved policy')

    args = parser.parse_args()
    print(" ")
    print("--fast-lr: {}".format(args.fast_lr))
    print(" ")
    # on my laptop: mp.cpu_count() - 1 = 3

    # Create logs and saves folder if they don't exist
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    if not os.path.exists('./saves'):
        os.makedirs('./saves')
    # Device
    args.device = torch.device(args.device
        if torch.cuda.is_available() else 'cpu')
    # Slurm
    if 'SLURM_JOB_ID' in os.environ:
        args.output_folder += '-{0}'.format(os.environ['SLURM_JOB_ID'])

    main(args)










