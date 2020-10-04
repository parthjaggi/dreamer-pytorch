#!/bin/bash
#SBATCH --account=def-ssanner
#SBATCH --gres=gpu:1           # Number of GPUs (per node)
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G              # memory (per node)
#SBATCH --time=0-24:00         # time (DD-HH:MM)

#SBATCH --mail-user=parthjaggi@iitrpr.ac.in
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

source ~/env/traffic4/bin/activate

# env=$1
# config=$2
# override=$3

# Can run following example commands. To be run from the sow45_code directory.
# sbatch run.sh
# sbatch wolf/scripts/compute_canada/run.sh test4_1 iql_global_reward_no_dueling_progression.yaml wolf/tests/override_configs/parth.yaml
# sbatch wolf/scripts/compute_canada/run.sh test4_1 iql_global_reward_noop.yaml wolf/tests/override_configs/parth.yaml
# sbatch wolf/scripts/compute_canada/run.sh test4_1 iql_global_reward_phase_select.yaml wolf/tests/override_configs/parth.yaml

python main_traffic.py --cuda-idx 0 --env-config-path /home/parthj/projects/def-ssanner/parthj/repos/sow45_code/wolf/tests/traffic_env/test4_2/iql_global_reward_no_dueling_image.yaml