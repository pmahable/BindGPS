#!/usr/bin/env python3
"""
Simple GPS Sweep Runner with WandB integration
"""
from itertools import product
from agent_trainer_classification_v2 import Config, GNNTrainer
from sweep_configs import get_sweep_config
import wandb

def train():
    """
    Training function called by wandb sweep agent.
    Gets hyperparameters from wandb.config and runs training.
    """
    # Initialize wandb run (required even in sweep mode)
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


def run_single(config_overrides=None, use_wandb=True):
    """Run a single experiment"""
    config = Config()
    
    # Apply any overrides
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # Set wandb settings
    config.use_wandb = use_wandb
    config.wandb_project = "basic-intro"
    config.wandb_entity = "bind-gps"
    config.use_wandb = False
    
    # Initialize wandb if requested
    if use_wandb:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            config=vars(config)
        )
    
    trainer = GNNTrainer(config)
    trainer.run()
    
    test_acc = trainer.evaluate()
    
    if use_wandb:
        wandb.finish()
    
    return test_acc


def run_wandb_sweep(sweep_config, count=10):
    """
    Run a wandb sweep
    
    Args:
        sweep_config: wandb sweep configuration dict
        count: number of runs to execute
    """
    # Create sweep
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        entity="bind-gps",
        project=sweep_config.get('project', 'basic-intro'),
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


def run_quick_test_with_models(count=4):
    """Run quick test with both GCN and GAT models"""
    from sweep_configs import get_sweep_config
    config = get_sweep_config('quick_test')
    run_wandb_sweep(config, count=count)


if __name__ == "__main__":
    # Run single experiment with GCN (default)
    # print("Running single GCN experiment...")
    # run_single()
    
    # Run single experiment with GAT
    print("Running single GAT experiment...")
    gat_test_config = get_sweep_config('quick_test')
    run_wandb_sweep(gat_test_config, count=1)
    
    # Example sweeps (uncomment to run):
    # print("Running quick test with both models...")
    # run_quick_test_with_models(count=4)
