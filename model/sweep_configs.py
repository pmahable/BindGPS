"""
Predefined WandB sweep configurations for BindGPS project.
Each config includes a project name for organized logging.
"""

# Quick test sweep (for debugging)
quick_test_config = {
    'method': 'grid',
    'metric': {'name': 'final_test_accuracy', 'goal': 'maximize'},
    'project': 'basic-gnn-sweep=v2',
    'parameters': {
        'lr': {'values': [5e-4]},
        'hidden_gnn_size': {'values': [64]},
        'model_type': {'values': ['gcn']},
        'epochs': {'value': 25}
    }
}

gat_test_dry_run = {
    'method': 'grid',
    'metric': {'name': 'final_test_accuracy', 'goal': 'maximize'},
    'project': 'gps-gat-test-dry-run',
    'parameters': {
        'lr': {'values': [1e-3]},
        'hidden_gnn_size': {'values': [64]},
        'model_type': {'values': ['gat']},
        'num_gnn_layers': {'values': [3]},
        'num_neighbors': {'values': [(20, 20, 20)]},
        'epochs': {'value': 25},
        # gat parameters
        'gat_heads': {'values': [2]},
        'gat_concat': {'values': [True]},
        'gat_negative_slope': {'values': [0.2]},
        'gat_edge_dim': {'values': [2]},
    }
}   

full_sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'final_test_accuracy', 'goal': 'maximize'},
    'project': 'gps-full-sweep',
    'parameters': {
        'lr': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1e-2},
        'weight_decay': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1e-2},
        'model_type': {'values': ['gcn', 'gat']},
        'hidden_gnn_size': {'values': [64, 128, 256, 512]},
        'num_gnn_layers': {'values': [2, 3, 4, 5]},
        'num_neighbors': {'values': [(20, 20), (20, 20, 20)]},
        'hidden_linear_size': {'values': [64, 128, 256, 512]},
        'num_linear_layers': {'values': [2, 3, 4]},
        'dropout': {'min': 0.1, 'max': 0.8},
        'batch_size': {'values': [128, 256, 512]},
        # GAT-specific parameters
        'gat_heads': {'values': [2, 4, 8]},
        'gat_concat': {'values': [True, False]},
        'gat_negative_slope': {'values': [0.1, 0.2, 0.3]},
        'epochs': {'value': 100}
    }
}

SWEEP_CONFIGS = {
    'quick_test': quick_test_config,
    'gat_test_dry_run': gat_test_dry_run,
    'full_sweep': full_sweep_config
}


def get_sweep_config(name):
    """Get a sweep configuration by name"""
    if name not in SWEEP_CONFIGS:
        available = list(SWEEP_CONFIGS.keys())
        raise ValueError(f"Unknown sweep config '{name}'. Available: {available}")
    return SWEEP_CONFIGS[name]
