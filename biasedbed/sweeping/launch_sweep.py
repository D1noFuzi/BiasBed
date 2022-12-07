import os
import wandb
from omegaconf import OmegaConf

from biasedbed.training.train import Training

cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), '..', 'config.yaml'))

def launch():
    train = Training(config=cfg)
    print('Start sweep')
    train.launch_training()

if os.getenv('SWEEP_ID'):
    wandb.agent(os.getenv('SWEEP_ID'), function=launch, count=1, project=cfg.project)
else:
    raise EnvironmentError('SWEEP_ID not set.')