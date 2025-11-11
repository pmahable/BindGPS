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

gat_model_no_svm_parameter_first_sweep = {
    #1A variables
    'method': 'grid', #use baysian when tuning 1C variables
    'metric': {'name': 'final_test_accuracy', 'goal': 'maximize'},
    'project': 'gps-gat-model-no-svm-parameter-test',
    'parameters': {
        'model_type': {'values': ['gat']},
        
        #parameters to tune
        'hidden_gnn_size': {'values': [64, 128, 256]},
        'hidden_linear_size': {'values': [64, 128, 256]},       
        'num_linear_layers': {'values': [2, 3]},        
        'gat_heads': {'values': [2, 4, 8]},     

        #constant variables
        #model
        'num_gnn_layers': {'value': 3},
        'dropout': {'value': 0.3},
        #optimization
        'lr': {'value': 0.001},
        'weight_decay': {'value': 0.0005},
        'epochs': {'value': 50},
        #neighbor loader
        'num_neighbors': {'value': (20, 20)},
        'batch_size': {'value': 256},
        #gat-parameters
        'gat_concat': {'value': True},
        'gat_negative_slope': {'value': 0.2},
        'gat_edge_dim': {'values': [0]}, #no edge attributes
    }
}

gat_model_no_svm_parameter_second_sweep = {
    #1B variables
    'method': 'grid', #use baysian when tuning 1C variables
    'metric': {'name': 'final_test_accuracy', 'goal': 'maximize'},
    'project': 'gps-gat-model-no-svm-parameter-test2',
    'parameters': {
        'model_type': {'values': ['gat']},
        
        #parameters to tune 
        'num_gnn_layers': {'values': [2, 3, 4]},
        'num_neighbors': {'values': [(10, 10), (20, 20), (10, 10, 10), (20, 20, 20)]},  
        'gat_concat': {'values': [True, False]},

        #constant variables
        #model
        'hidden_gnn_size': {'values': [64]}, #best loss
        'hidden_linear_size': {'values': [128]}, #best loss
        'gat_heads': {'values': [4]}, #most information
        'num_linear_layers': {'values': [3]}, #unclear -- most information
        'dropout': {'value': 0.3},
        #optimization
        'lr': {'value': 0.001},
        'weight_decay': {'value': 0.0005},
        'epochs': {'value': 50},
        #neighbor loader
        'batch_size': {'value': 256},
        #gat-parameters
        'gat_negative_slope': {'value': 0.2},
        'gat_edge_dim': {'values': [0]}, #no edge attributes
    }
}

gat_model_no_svm_parameter_third_sweep = {
    #1 variables
    'method': 'bayes', #use baysian when tuning 1C variables
    'metric': {'name': 'final_test_accuracy', 'goal': 'maximize'},
    'project': 'gps-gat-model-no-svm-parameter-test3',
    'parameters': {
        'model_type': {'values': ['gat']},
        
        #parameters to tune 
        'lr': {'values': [0.01, 0.001, 0.0005]},
        'dropout': {'values': [0.2, 0.3, 0.4, 0.5]},
        
        #constant variables
        #model
        'hidden_gnn_size': {'values': [64]}, #best loss
        'hidden_linear_size': {'values': [128]}, #best loss
        'gat_heads': {'values': [4]}, #most information
        'num_linear_layers': {'values': [3]}, #unclear -- most information
        'num_gnn_layers': {'values': [3]}, #was between 2 and 3 layers
        #optimization
        'weight_decay': {'value': 0.0005},
        'epochs': {'value': 50},
        #neighbor loader
        'num_neighbors': {'values': [(20, 20, 20)]},
        'batch_size': {'value': 256},
        #gat-parameters
        'gat_concat': {'values': [True]},
        'gat_negative_slope': {'value': 0.2},
        'gat_edge_dim': {'values': [0]}, #no edge attributes
    }
}

gat_model_parameter_first_sweep = {
    'method': 'grid',  
    'metric': {'name': 'final_test_accuracy', 'goal': 'maximize'},
    'project': 'gps-gat-model-parameter-test',
    'parameters': {
        'model_type': {'values': ['gat']},
        'hidden_gnn_size': {'values': [64, 128, 256]},         # variable
        'num_linear_layers': {'values': [2, 3]},          # variable
        'gat_heads': {'values': [2, 4, 8]},                  # variable
        # keep these constant
        'lr': {'value': 0.001},
        'weight_decay': {'value': 0.0005},
        'dropout': {'value': 0.3},
        'batch_size': {'value': 256},
        'epochs': {'value': 50},
        'num_gnn_layers': {'value': 3},
        'num_neighbors': {'value': (20, 20)},
        'gat_concat': {'value': True},
        'gat_negative_slope': {'value': 0.2},
        'gat_edge_dim': {'values': [0]},
    }
}

gat_model_parameter_sweep_2 = {
    'method': 'grid',  
    'metric': {'name': 'final_test_accuracy', 'goal': 'maximize'},
    'project': 'gps-gat-model-parameter-test',
    'parameters': {
        'model_type': {'values': ['gat']},
        'hidden_gnn_size': {'values': [128, 256]},         #variable
        'hidden_linear_size': {'values': [64, 128, 256]}, #variable
        'num_linear_layers': {'values': [2, 4]},          # variable
        'gat_heads': {'values': [2, 4]},                  # variable
        # keep these constant
        'lr': {'value': 0.001},
        'weight_decay': {'value': 0.0005},
        'dropout': {'value': 0.3},
        'batch_size': {'value': 256},
        'epochs': {'value': 50},
        'num_gnn_layers': {'value': 3},
        'num_neighbors': {'value': (20, 20)},
        'gat_concat': {'value': True},
        'gat_negative_slope': {'value': 0.2},
        'gat_edge_dim': {'values': [0]},
    }
}

gat_data_parameter_sweep = {
    'method': 'grid',
    'metric': {'name': 'final_test_accuracy', 'goal': 'maximize'},
    'project': 'gps-gat-data-parameter-test',
    'parameters': {
        'model_type': {'value': 'gat'},
        # variables (data-related)
        'num_gnn_layers': {'values': [2, 3, 4]},                 
        'num_neighbors': {'values': [(10, 10), (20, 20), (10, 10, 10), (20, 20, 20)]},  
        'gat_concat': {'values': [True, False]},
        # fixed
        'gat_edge_dim': {'values': [0]}, 
        'hidden_gnn_size': {'value': 256},
        'gat_heads': {'value': 2},
        'num_linear_layers': {'value': 2},
        'hidden_linear_size': {'value': 128},
        'gat_negative_slope': {'value': 0.2},
        'dropout': {'value': 0.3},
        'lr': {'value': 0.001},
        'weight_decay': {'value': 0.0005},
        'batch_size': {'value': 256},
        'epochs': {'value': 35},
    }
}
    
SWEEP_CONFIGS = {
    'quick_test': quick_test_config,
    'gat_test_dry_run': gat_test_dry_run,
    'full_sweep': full_sweep_config,
    #GAT Parameter Sweeps - ROMER
    'gat_model_param_sweep': gat_model_parameter_first_sweep,
    'gat_model_param_sweep_2': gat_model_parameter_sweep_2,
    'gat_data_param_sweep': gat_data_parameter_sweep,
    #GAT Parameter Sweeps - SARAH 
    'gat_model_no_svm_param_sweep': gat_model_no_svm_parameter_first_sweep,
    'gat_model_no_svm_param_sweep2': gat_model_no_svm_parameter_second_sweep, 
    'gat_model_no_svm_param_sweep3': gat_model_no_svm_parameter_third_sweep
}

def get_sweep_config(name):
    """Get a sweep configuration by name"""
    if name not in SWEEP_CONFIGS:
        available = list(SWEEP_CONFIGS.keys())
        raise ValueError(f"Unknown sweep config '{name}'. Available: {available}")
    return SWEEP_CONFIGS[name]
