import wandb
from typing import Dict, Any, Optional, Callable
from contextlib import contextmanager

class WandbLogger:
    """
    Simple wandb logging utility for the BindGPS project.
    Two main functionalities: single experiments and sweeps.
    """
    
    def __init__(self, entity: str = "bind-gps", project: str = "basic-intro"):
        """
        Initialize the WandbLogger.
        
        Args:
            entity: Wandb entity name (default: "bind-gps")
            project: Wandb project name (default: "basic-intro")
        """
        self.entity = entity
        self.project = project
        self.current_run = None
        
        # Ensure wandb is logged in
        try:
            wandb.login()
        except:
            print("Please login to wandb first: wandb.login()")
    
    @contextmanager
    def experiment(self, name: str, config: Dict[str, Any]):
        """
        Run a single experiment.
        
        Args:
            name: Experiment name
            config: Configuration dictionary for hyperparameters
        
        Usage:
            logger = WandbLogger()
            with logger.experiment("my_experiment", {"lr": 0.01}) as run:
                # Your training code
                logger.log({"loss": 0.5, "accuracy": 0.95})
        """
        try:
            self.current_run = wandb.init(
                entity=self.entity,
                project=self.project,
                name=name,
                config=config
            )
            yield self.current_run
        finally:
            if self.current_run:
                wandb.finish()
                self.current_run = None
    
    def log(self, metrics: Dict[str, float]):
        """
        Log metrics to wandb.
        
        Args:
            metrics: Dictionary of metric names and values
        """
        if self.current_run:
            wandb.log(metrics)
        else:
            print("No active run. Start an experiment first.")
    
    def run_sweep(self, sweep_config: Dict[str, Any], train_function: Callable, count: int = 10):
        """
        Create and run a hyperparameter sweep.
        
        Args:
            sweep_config: Sweep configuration dictionary
            train_function: Function that contains your training logic
            count: Number of runs to execute
            
        Usage:
            def train():
                with logger.sweep_run():
                    lr = wandb.config.learning_rate
                    # Your training code
                    logger.log({"val_loss": loss})
            
            sweep_config = {
                'method': 'random',
                'metric': {'name': 'val_loss', 'goal': 'minimize'},
                'parameters': {
                    'learning_rate': {'values': [0.001, 0.01, 0.1]}
                }
            }
            logger.run_sweep(sweep_config, train, count=5)
        """
        # Create sweep
        sweep_id = wandb.sweep(
            sweep=sweep_config,
            entity=self.entity,
            project=self.project
        )
        
        print(f"Created sweep: {sweep_id}")
        
        # Run sweep agent
        wandb.agent(
            sweep_id=sweep_id,
            function=train_function,
            entity=self.entity,
            project=self.project,
            count=count
        )
    
    @contextmanager
    def sweep_run(self):
        """
        Context manager for individual sweep runs.
        Use this inside your train function when running sweeps.
        """
        try:
            self.current_run = wandb.init()
            yield self.current_run
        finally:
            if self.current_run:
                wandb.finish()
                self.current_run = None
