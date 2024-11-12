#!/bin/bash
set -e

export OMP_NUM_THREADS=8  # nb_cpu_threads / nproc_per_node
export PYTHONPATH=$(builtin cd ..; pwd)

cd ../../
. main.config
export WANDB_ENTITY=${WANDB_ENTITY}
export WANDB_PROJECT=${WANDB_PROJECT}

export DS_SKIP_CUDA_CHECK=1

DATASET=medqa_usmle
SEED=0

## Graph
EMBEDS_ORIGIN=umls

## LM
model="mistralai/Mistral-7B-Instruct-v0.1"
modelname=meg-${MODELS[$model]}
echo ${model} ${modelname}

LR=1e-4
MAX_NEW_TOKENS=30
MODEL_MAX_LEN=1024
LORA_R=32
LORA_ALPHA=64

## Logs
WANDB_NAME=${DATASET}-graph_in_prompt-${modelname}-lr${LR}-${MODEL_MAX_LEN}-${EMBEDS_ORIGIN}-s${SEED}

cd ${TRAIN_CODE_DIR}

#############
##  TRAIN  ##
#############
logging_file=${LOGS_DIR}/logs_${DATASET}/logs_${WANDB_NAME}.txt
echo "Logging to ${logging_file}"

deepspeed --include localhost:0,1,2,3 --master_port 29502 train_meg_mistral.py > ${logging_file} 2>&1  \
--deepspeed ds/deepspeed_config_s2.json \
--task ${DATASET} \
--model_name_or_path ${model} \
--modelname ${modelname} \
--data_path ${ROOT_DATA_DIR}/${DATASET}/train_with_graph_embeds_no_marker_end.jsonl \
--test_data_path ${ROOT_DATA_DIR}/${DATASET}/test_with_graph_embeds_no_marker_end.jsonl \
--graph_in_prompt ${ROOT_DATA_DIR}/${EMBEDS_ORIGIN}/neighbors_data_n25.jsonl \
--output_dir ${CKPT_DIR}/${WANDB_NAME} \
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
--logging_steps 50 \
--save_total_limit 1 \
--learning_rate ${LR} \
--weight_decay 0.     \
--warmup_ratio 0.03    \
--loss_temperature ${loss_temp} \
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

out_ckpt=${CKPT_DIR}/${WANDB_NAME}/checkpoint_last
echo "Running inference for ${out_ckpt}"
output_file=${out_ckpt}/test_answers.jsonl

logging_file=${LOGS_DIR}/logs_${DATASET}/logs_eval_${WANDB_NAME}.txt
echo "Logging to ${logging_file}"

deepspeed --include localhost:0,1 --master_port 29501 eval_meg_mistral.py > ${logging_file} 2>&1 \
--deepspeed ds/deepspeed_config_s2.json \
--task ${DATASET} \
--mapping_type ${mapping_type} \
--model_name_or_path ${model} \
--modelname ${modelname} \
--resume_from_checkpoint ${out_ckpt} \
--test_data_path ${ROOT_DATA_DIR}/${DATASET}/test_with_graph_embeds_no_marker_end.jsonl \
--graph_in_prompt ${ROOT_DATA_DIR}/${EMBEDS_ORIGIN}/neighbors_data_n25.jsonl \
--output_dir ${output_file} \
--padding_side left \
--bf16 True \
--num_train_epochs 3 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 8 \
--eval_accumulation_steps 2 \
--save_strategy "epoch"  \
--save_steps 1 \
--logging_strategy "steps" \
--logging_steps 50 \
--save_total_limit 1 \
--learning_rate ${LR} \
--weight_decay 0.     \
--warmup_ratio 0.03    \
--loss_temperature ${loss_temp} \
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
