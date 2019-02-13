#!/usr/bin/env python3
import argparse
import multiprocessing
import os
import sys

import tensorflow as tf

import gym
from baselines import logger
from baselines.common import set_global_seeds
from baselines.common.cmd_util import make_multiple_mujoco_env
from baselines.common.vec_env.vec_normalize import ImVecNormalize
from baselines.flow_rl_utils.network import flow_dict
from baselines.ppo2 import ppo2
from baselines.ppo2.policies import MImVecPolicy, MImVecLstmPolicy, MImVecLnLstmPolicy


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


class EnvParameterChanger():
    def get_gym_env_parameter(self, parser):
        tmp_args, _ = parser.parse_known_args()
        self.env = tmp_args.env_id
        assert self.env in gym.envs.registry.env_specs
        self.env_args_dict = gym.envs.registry.env_specs[self.env]._kwargs
        for key, value in self.env_args_dict.items():
            if key not in tmp_args:
                parser.add_argument('--' + key,
                                    type=type(value) if type(value) != bool else str2bool,
                                    default=value)
        return parser

    def set_gym_env_parameter(self, args):
        args_dict = vars(args)
        for key in self.env_args_dict:
            gym.envs.registry.env_specs[self.env]._kwargs[key] = args_dict[key]


def train(args):
    logger.configure(args.main_path)

    if args.diff_frames:
        assert "stack_frames" in args

    seed = int.from_bytes(os.urandom(4), byteorder='big')
    set_global_seeds(seed)
    env = ImVecNormalize(make_multiple_mujoco_env(
        args.env_id, args.number_of_agents, seed))

    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True
    tf.Session(config=config).__enter__()
    with tf.device("/device:GPU:0"):
        if args.policy == "cnn":
            policy = MImVecPolicy
        elif args.policy == "lstm_cnn":
            policy = MImVecLstmPolicy
        elif args.policy == "lnlstm_cnn":
            policy = MImVecLnLstmPolicy
        else:
            raise ValueError

    ppo2.learn(
        policy=policy,
        env=env,
        nsteps=args.nsteps,
        nminibatches=args.nminibatches,
        lam=args.lam,
        gamma=args.gamma,
        noptepochs=args.noptepochs,
        log_interval=1,
        ent_coef=0.0,
        lr=args.learning_rate,
        cliprange=args.cliprange,
        total_timesteps=int(args.num_timesteps * 1.01),
        add_flownet=args.add_flownet,
        flownet_path=args.flownet_path,
        flow_key=args.flow_key,
        train_from_scratch=args.train_from_scratch,
        large_cnn=args.large_cnn,
        add_predicted_flow_to_vec=args.add_predicted_flow_to_vec,
        diff_frames=args.diff_frames)


def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser()

        parser.add_argument('--main_path', type=str, default=None)
        parser.add_argument('--env_id', type=str, default="Chaser2d-v2")
        parser.add_argument('--nsteps', type=int, default=128)
        parser.add_argument('--nminibatches', type=int, default=4)
        parser.add_argument('--lam', type=float, default=0.95)
        parser.add_argument('--gamma', type=float, default=0.99)
        parser.add_argument('--noptepochs', type=int, default=2)
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        parser.add_argument('--cliprange', type=float, default=0.2)
        parser.add_argument('--num_timesteps', type=int, default=int(2e7))
        parser.add_argument('--number_of_agents', type=int, default=8)
        parser.add_argument('--add_flownet', type=str2bool, default=False)
        parser.add_argument('--flownet_path', type=str, default="")
        parser.add_argument('--flow_key', type=str, default='normal', choices=flow_dict)
        parser.add_argument('--policy', type=str, choices=("cnn", "lstm_cnn", "lnlstm_cnn"),
                            default="cnn")
        parser.add_argument('--train_from_scratch', type=str2bool, default=False)
        parser.add_argument('--large_cnn', type=str2bool, default=False)
        parser.add_argument('--add_predicted_flow_to_vec', type=str2bool, default=False)
        parser.add_argument('--diff_frames', type=str2bool, default=False)

        changer = EnvParameterChanger()
        parser = changer.get_gym_env_parameter(parser)

        args = parser.parse_args()
        changer.set_gym_env_parameter(args)

    train(args)

if __name__ == '__main__':
    main()
