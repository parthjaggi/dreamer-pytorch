#!/bin/bash -l

# Specify hard time limit for the job.
#   The job will be aborted if it runs longer than this time.
#   The default time is 12 hours
#$ -l h_rt=24:00:00

# Send an email when the job finishes or if it is aborted (by default no email $
#$ -m bea

#$ -M kasra0gh@bu.edu

# Give job a name
#$ -N rlpyt_example

# Request 4 CPUs
#$ -pe omp 4

# Request 1 GPU (the number of GPUs needed should be divided by the number of C$
#$ -l gpus=0.25

# Specify the minimum GPU compute capability
#$ -l gpu_c=3.5

# Keep track of information related to the current job
echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID"
echo "=========================================================="

module load cuda/10.1
module load python3/3.6.9
module load pytorch/1.3

python3 example_1.py

