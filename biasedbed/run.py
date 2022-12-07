import os
import argparse

from omegaconf import OmegaConf
from biasedbed.training.train import Training
from biasedbed.sweeping.sweep import Sweep

from biasedbed.sweeping.launcher import bsub_launcher, basic_launcher


parser = argparse.ArgumentParser()
parser.add_argument('command', choices=['single', 'sweep'], help='Train single network or start hyperparameter sweep')
parser.add_argument('--data_dir', type=str, help='Root directory of datasets.')
parser.add_argument('--algorithm', type=str, help='Trainings algorithm.')
parser.add_argument('--count', type=int, default=3, help='Number of (random) sweeps to start.')
args = parser.parse_args()

# Load config file
cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), 'config.yaml'))
if args.data_dir is not None:
    OmegaConf.update(cfg, 'data_dir', args.data_dir)
if args.algorithm is not None:
    OmegaConf.update(cfg, 'algorithm.name', args.algorithm)


def launch():
    train = Training(config=cfg)
    train.launch_training()


def main():
    if args.command == 'single':
        print('Starting single run')
        launch()

    elif args.command == 'sweep':
        print('Starting sweep')
        sweep = Sweep(config=cfg)
        sweep_id = sweep.init_sweep()
        for i in range(args.count):
            # Pass sweep id to agents running on cluster node
            # bsub_launcher(cmd='python launch_sweep.py', sweep_id=sweep_id)
            basic_launcher(f"SWEEP_ID={sweep_id} python sweeping/launch_sweep.py")


if __name__ == '__main__':
    main()
