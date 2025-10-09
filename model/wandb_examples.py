"""
Simple examples of using WandbLogger for the BindGPS project.
Two main functionalities: single experiments and sweeps.
"""

import random
from src.wandb_utils import WandbLogger
import wandb

# Example 1: Single Experiment
def example_single_experiment():
    """Example of running a single experiment with logging."""
    
    logger = WandbLogger(entity="bind-gps", project="basic-intro")
    
    # Configuration for this experiment
    config = {
        "learning_rate": 0.01,
        "batch_size": 32,
        "epochs": 10,
        "model_type": "CNN"
    }
    
    # Run experiment
    with logger.experiment("my_experiment", config) as run:
        print(f"Running experiment with config: {config}")
        
        # Simulate training loop
        for epoch in range(config["epochs"]):
            # Simulate training metrics
            train_loss = 1.0 - (epoch / config["epochs"]) * 0.8 + random.random() * 0.1
            val_loss = train_loss + random.random() * 0.05
            accuracy = 1.0 - val_loss + random.random() * 0.1
            
            # Log metrics
            logger.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "accuracy": accuracy
            })
        
        print("Experiment completed!")


# Example 2: Hyperparameter Sweep
def example_sweep():
    """Example of running a hyperparameter sweep."""
    
    logger = WandbLogger(entity="bind-gps", project="basic-intro")
    
    # Define sweep configuration
    sweep_config = {
        'method': 'random',
        'metric': {
            'name': 'val_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'learning_rate': {
                'values': [0.001, 0.01, 0.1]
            },
            'batch_size': {
                'values': [16, 32, 64]
            },
            'dropout': {
                'values': [0.1, 0.2, 0.3]
            }
        }
    }
    
    # Training function for sweep
    def train():
        with logger.sweep_run():
            # Get hyperparameters from wandb.config
            lr = wandb.config.learning_rate
            batch_size = wandb.config.batch_size
            dropout = wandb.config.dropout
            
            print(f"Training with lr={lr}, batch_size={batch_size}, dropout={dropout}")
            
            # Simulate training
            for epoch in range(5):
                # Simulate metrics that depend on hyperparameters
                train_loss = random.random() * (1 / lr) * (dropout + 0.1)
                val_loss = train_loss + random.random() * 0.1
                accuracy = max(0, min(1, 1 - val_loss))
                
                logger.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "accuracy": accuracy
                })
    
    # Run sweep with 5 experiments
    logger.run_sweep(sweep_config, train, count=5)
    print("Sweep completed!")


if __name__ == "__main__":
    print("WandbLogger Examples")
    print("1. Single experiment")
    print("2. Hyperparameter sweep")
    
    choice = input("Choose example (1 or 2): ")
    
    if choice == "1":
        example_single_experiment()
    elif choice == "2":
        example_sweep()
    else:
        print("Invalid choice")
