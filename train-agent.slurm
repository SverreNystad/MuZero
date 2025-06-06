#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name="muzero-car-racing"
#SBATCH --time=3-00:25:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --nodes=1
#SBATCH --constraint="(v100|a100|h100)&(gpu32g|gpu40g|gpu80g)"
#SBATCH --output=slurm_outputs/output_combined.txt
#SBATCH --error=slurm_outputs/output_combined.err
#SBATCH --mail-user=sverrnys@ntnu.no
#SBATCH --mail-type=ALL


WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

ENV_PATH="/cluster/work/$USER/muzero/muzero-env/"

module purge
module load Anaconda3/2024.02-1

# Create a new Conda environment
conda create -y --prefix ${ENV_PATH} python=3.11

# Activate the environment
conda activate ${ENV_PATH}
pip install -r requirements.txt

echo "Downloaded dependencies:"
pip freeze

# Run the Python script
python main.py

# Deactivate the environment
conda deactivate

echo "Job finished"


# "sbatch train-agent.slurm"  to run the job
