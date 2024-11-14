#!/bin/bash
set -e

export OMP_NUM_THREADS=8  # nb_cpu_threads / nproc_per_node
export PYTHONPATH=$(builtin cd ..; pwd)

cd ../../
. main.config
export WANDB_ENTITY=${WANDB_ENTITY}
export WANDB_PROJECT=${WANDB_PROJECT}

export DS_SKIP_CUDA_CHECK=1

DATASET=pubmedqa
SEED=0

## LM
model="meta-llama/Llama-3.1-8B-Instruct"
modelname=${MODELS[$model]}
echo ${model} ${modelname}

LR=1e-4
MAX_NEW_TOKENS=30
MODEL_MAX_LEN=500
LORA_R=32
LORA_ALPHA=64

## Logs
WANDB_NAME=${DATASET}-${modelname}-lr${LR}-${MODEL_MAX_LEN}-s${SEED}
cd ${TRAIN_CODE_DIR}

#############
##  TRAIN  ##
#############
logging_file=${LOGS_DIR}/logs_${DATASET}/logs_${WANDB_NAME}.txt
echo "Logging to ${logging_file}"

torchrun --nproc_per_node=4 --master_port=9779 train_meg_llama.py > ${logging_file} 2>&1  \
--task ${DATASET} \
--model_name_or_path ${model} \
--modelname ${modelname} \
--data_path ${BASE_DATA_DIR}/${DATASET}/train_with_graph_embeds_no_marker_end.jsonl \
--test_data_path ${BASE_DATA_DIR}/${DATASET}/test_with_graph_embeds_no_marker_end.jsonl \
--output_dir ${CKPT_DIR}/${DATASET}/${WANDB_NAME} \
--padding_side left \
--bf16 True \
--num_train_epochs 3 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 8 \
--eval_accumulation_steps 1 \
--save_strategy "epoch"  \
--save_steps 1 \
--logging_strategy "steps" \
--logging_steps 5 \
--save_total_limit 1 \
--learning_rate ${LR} \
--weight_decay 0.     \
--warmup_ratio 0.03    \
--lr_scheduler_type "cosine"   \
--model_max_length ${MODEL_MAX_LEN} \
--max_new_tokens ${MAX_NEW_TOKENS} \
--q_lora False \
--lora_r ${LORA_R} \
--lora_alpha ${LORA_ALPHA} \
--lora_target_modules 'q_proj' 'k_proj' 'v_proj' 'o_proj' 'gate_proj' 'up_proj' 'down_proj' \
--gradient_checkpointing False \
--temp 0.2 \
--seed ${SEED} \
--run_name ${WANDB_NAME} \
--report_to "wandb"


############
##  EVAL  ##
############

out_ckpt=${CKPT_DIR}/${DATASET}/${WANDB_NAME}/checkpoint_last
echo "Running inference for ${out_ckpt}"
output_file=${out_ckpt}/test_answers.jsonl

logging_file=${LOGS_DIR}/logs_${DATASET}/logs_eval_${WANDB_NAME}.txt
echo "Logging to ${logging_file}"

torchrun --nproc_per_node=1 --master_port=9779 eval_meg_llama.py > ${logging_file} 2>&1 \
--task ${DATASET} \
--model_name_or_path ${model} \
--modelname ${modelname} \
--test_data_path ${BASE_DATA_DIR}/${DATASET}/test_with_graph_embeds_no_marker_end.jsonl \
--resume_from_checkpoint ${CKPT_DIR}/${DATASET}/${WANDB_NAME}/checkpoint_last \
--output_dir ${CKPT_DIR}/${DATASET}/${WANDB_NAME} \
--padding_side right \
--bf16 False \
--num_train_epochs 3 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 1 \
--eval_accumulation_steps 2 \
--save_strategy "epoch"  \
--save_steps 1 \
--logging_strategy "steps" \
--logging_steps 5 \
--save_total_limit 1 \
--learning_rate ${LR} \
--weight_decay 0.     \
--warmup_ratio 0.03    \
--lr_scheduler_type "cosine" \
--model_max_length ${MODEL_MAX_LEN} \
--max_new_tokens ${MAX_NEW_TOKENS} \
--q_lora False \
--lora_r ${LORA_R} \
--lora_alpha ${LORA_ALPHA} \
--lora_target_modules 'q_proj' 'k_proj' 'v_proj' 'o_proj' 'gate_proj' 'up_proj' 'down_proj' \
--gradient_checkpointing False \
--temp 0.2 \
--seed ${SEED} \
--freeze 

echo "Output file: ${output_file}"
