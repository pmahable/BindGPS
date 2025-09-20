

## Setup and Permission Management
To set up the environment, run the following commands:
```bash
# enter an interact session (don't run on login node)
interact -m 64g

python -m venv ~/bind_gps_env
source ~/bind_gps_env/bin/activate

python -m pip install torch numpy pandas scipy
python -m pip install torch-scatter torch-sparse torch-geometric
python -m pip install cooler pybedtools

# linting
python -m pip install black isort
```

To make sure all files are accessible by group members run: 
```bash
make permission
```