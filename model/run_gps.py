#!/usr/bin/env python3
"""
Simple GPS Sweep Runner with WandB integration
"""
from itertools import product
from agent_trainer_classification_v2 import Config, GNNTrainer

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def train():
    """
    Training function called by wandb sweep agent.
    Gets hyperparameters from wandb.config and runs training.
    """

    # Initialize wandb run (done automatically by sweep agent)
    wandb.init()
    config = Config()

    # Override config with wandb sweep parameters
    for key, value in wandb.config.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Ensure wandb is enabled and properly configured
    config.use_wandb = True
    config.wandb_project = wandb.run.project
    config.wandb_entity = wandb.run.entity
    
    # Run training
    trainer = GNNTrainer(config)
    trainer.run()
    
    return


def run_single(config_overrides=None):
    """Run a single experiment"""
    config = Config()
    
    # Apply any overrides
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # Set wandb settings
    config.use_wandb = WANDB_AVAILABLE
    config.wandb_project = "basic-gnn"
    config.wandb_entity = "bind-gps"
    config.use_wandb = False
    
    trainer = GNNTrainer(config)
    trainer.run()
    
    return


def run_wandb_sweep(sweep_config, count=10):
    """
    Run a wandb sweep
    
    Args:
        sweep_config: wandb sweep configuration dict
        count: number of runs to execute
    """
    if not WANDB_AVAILABLE:
        raise RuntimeError("wandb is required for wandb sweeps")

    # Create sweep
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        entity="bind-gps",
        project=sweep_config.get('project', 'basic-gnn-sweep')
    )
    
    print(f"Created sweep: {sweep_id}")
    print(f"Starting {count} runs...")
    
    # Run sweep agent
    wandb.agent(
        sweep_id=sweep_id,
        function=train,
        entity="bind-gps",
        project=sweep_config.get('project', 'basic-gnn-sweep'),
        count=count
    )

if __name__ == "__main__":
    # Run single experiment
    # print("Running single experiment...")
    # run_single()
    
    # Example wandb sweep (uncomment to run):
    from sweep_configs import quick_test_config
    run_wandb_sweep(quick_test_config, count=5)