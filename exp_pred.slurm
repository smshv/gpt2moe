#!/bin/bash

#SBATCH -A shengroup
#SBATCH --partition=gpu
#SBATCH --mem=32768
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=20:00:00
### END OF SLURM PARAMS ##

epochs=10
train_max_step=20000
test_max_step=2000

NUM_NODES=1
NUM_GPUS=1

module purge
module load anaconda/2023.07-py3.11
source activate deepspeed
module --ignore_cache load "gcc/11.4.0"
module load cuda/11.4.2

#cp exp_pred_ds/layer_orig.py ~/env_ds_site-packages/deepspeed/moe/layer.py
#cp exp_pred_ds/experts_orig.py ~/env_ds_site-packages/deepspeed/moe/experts.py
#cp exp_pred_ds/sharded_moe_orig.py ~/env_ds_site-packages/deepspeed/moe/sharded_moe.py

cp ~/env_ds_site-packages/deepspeed/moe/layer.py exp_pred_ds/layer_orig.py
cp ~/env_ds_site-packages/deepspeed/moe/experts.py exp_pred_ds/experts_orig.py
cp ~/env_ds_site-packages/deepspeed/moe/sharded_moe.py exp_pred_ds/sharded_moe_orig.py

cp exp_pred_ds/layer.py ~/env_ds_site-packages/deepspeed/moe/layer.py
cp exp_pred_ds/experts.py ~/env_ds_site-packages/deepspeed/moe/experts.py
cp exp_pred_ds/sharded_moe.py ~/env_ds_site-packages/deepspeed/moe/sharded_moe.py

deepspeed --num_node=$NUM_NODES --num_gpus=$NUM_GPUS exp_pred.py \
	--epochs $epochs \
	--train_max_step $train_max_step \
	--test_max_step $test_max_step

mv exp_pred_ds/layer_orig.py ~/env_ds_site-packages/deepspeed/moe/layer.py
mv exp_pred_ds/experts_orig.py ~/env_ds_site-packages/deepspeed/moe/experts.py
mv exp_pred_ds/sharded_moe_orig.py ~/env_ds_site-packages/deepspeed/moe/sharded_moe.py
