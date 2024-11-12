#!/bin/bash
# set -e

export OMP_NUM_THREADS=8  # nb_cpu_threads / nproc_per_node
export PYTHONPATH=$(builtin cd ..; pwd)

cd ../../
. main.config
export WANDB_ENTITY=${WANDB_ENTITY}
export WANDB_PROJECT=${WANDB_PROJECT}

export DS_SKIP_CUDA_CHECK=1

DATASET=umls
SEED=0

## Graph
EMBEDS_ORIGIN=umls
EMBEDS_FILE=word2embed_graphsage_sapbertinit.h5
EMBED_DIM=256

## LM
model="mistralai/Mistral-7B-Instruct-v0.1"
modelname=meg-${MODELS[$model]}
echo ${model} ${modelname}

LR=1e-5
MAX_NEW_TOKENS=20
MODEL_MAX_LEN=200 
LORA_R=32
LORA_ALPHA=64

## Mapping 
mapping_type="mlp"
input_prefix_length=1
output_prefix_length=1
num_layers=4
num_heads=1
loss_temp=1.0

## Logs
WANDB_NAME=${DATASET}-${modelname}-lr${LR}-${MODEL_MAX_LEN}-${mapping_type}-${EMBED_DIM}-${input_prefix_length}-${output_prefix_length}-${EMBEDS_ORIGIN}-s${SEED}-extended
logging_file=${LOGS_DIR}/logs_${DATASET}/logs_${WANDB_NAME}.txt
echo "Logging to ${logging_file}"

## ==================================================================

cd ${TRAIN_CODE_DIR}

deepspeed --include localhost:0,1,2,3 --master_port 29502 train_meg_mistral.py > ${logging_file} 2>&1 \
--deepspeed ds/deepspeed_config_s2.json \
--task ${DATASET} \
--mapping_type ${mapping_type} \
--model_name_or_path ${model} \
--modelname ${modelname} \
--data_path ${ROOT_DATA_DIR}/${EMBEDS_ORIGIN}/embeddings/vocab_training_meg_extended.jsonl \
--embeddings_dir ${ROOT_DATA_DIR}/${EMBEDS_ORIGIN}/embeddings/${EMBEDS_FILE} \
--output_dir ${CKPT_DIR}/${WANDB_NAME} \
--padding_side left \
--bf16 True \
--num_train_epochs 6 \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2 \
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
--compound_loss True \
--prefix_length ${input_prefix_length} \
--output_prefix_length ${output_prefix_length} \
--prefix_dim ${EMBED_DIM} \
--num_heads_map_network ${num_heads} \
--num_layers_map_network ${num_layers} \
--seed ${SEED} \
--run_name ${WANDB_NAME} \
--report_to "wandb"
