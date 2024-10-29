#!/bin/bash

# Check if running on a cluster (assuming SLURM)
if [ -n "$SLURM_JOB_ID" ]; then
    # Cluster-specific setup
    module load cuda python
    
    # Set GPU visibility
    export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
else
    # Local machine setup
    echo "Running on local machine"
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the training
python main.py $@
