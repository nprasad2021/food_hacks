#!/bin/bash
#SBATCH -n 4
#SBATCH --array=0-11
#SBATCH --job-name=food_hacks
#SBATCH --mem=10GB
#SBATCH --gres=gpu:tesla-k80:1
#SBATCH -t 03:00:00
#SBATCH --workdir=./subs/

cd ..
singularity exec -B /om:/om --nv /om/user/nprasad/singularity/belledon-tensorflow-keras-master-latest.simg \
python /om/user/nprasad/food_hacks/main.py ${SLURM_ARRAY_TASK_ID}