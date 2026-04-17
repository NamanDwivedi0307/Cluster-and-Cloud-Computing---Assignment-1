#!/bin/bash
#SBATCH --job-name=langcount
#SBATCH --time=03:00:00
#SBATCH --output=langcount_%j.out
#SBATCH --error=langcount_%j.err

module --force purge
module load foss/2022a
module load Python/3.10.4
module load mpi4py/3.1.4

cd ~/CCC_A1

echo "Job started on $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Number of tasks: $SLURM_NTASKS"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-1}"
echo "Tasks per node: ${SLURM_NTASKS_PER_NODE:-N/A}"
echo "Configuration summary: ${SLURM_JOB_NUM_NODES} node(s), ${SLURM_NTASKS} task(s), ${SLURM_CPUS_PER_TASK:-1} CPU(s) per task"

srun -n ${SLURM_NTASKS} python language_counter_mpi.py "$1" "$2"

echo "Job finished on $(date)"
