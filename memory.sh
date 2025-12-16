#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=24:00:00
#PBS -P gcc50527
#PBS -j oe
#PBS -m abe
#PBS -M shimomura.teruki174@mail.kyutech.jp


module purge
module load cuda/12.8
module load cudnn/9.10

source ~/miniconda3/etc/profile.d/conda.sh



export CUDA_VISIBLE_DEVICES=0
export CC=gcc
export CXX=g++


cd ${PBS_O_WORKDIR}



conda activate ais
nvidia-smi
python3 memory.py

