#! /bin/sh
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gpus=geforce:1
#SBATCH --time=00:01:00
#SBATCH --job-name=deviceQuery
#SBATCH --output=/blue/cis4936/d.bodra/part1/deviceQuery.log
pwd; hostname; date
module load ufrc cmake/3.19.1 intel/2018.1.163 cuda/10.0.130 git
echo "Running on GPU"
nvcc ./deviceQuery.cu -o deviceQuery
srun ./deviceQuery