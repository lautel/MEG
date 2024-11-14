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
# EMBEDS_FILE=word2embed_rdf2vec_50epochs.h5
# EMBEDS_FILE=word2embed_edgegraphsage_sapbertinit.h5
EMBEDS_FILE=word2embed_graphsage_sapbertinit.h5
EMBED_DIM=256
# EMBEDS_FILE=word2embed_distmult.h5
# EMBED_DIM=50
# EMBEDS_FILE=word2embed_sapbert.h5
# EMBED_DIM=768

## LM
model="mistralai/Mistral-7B-Instruct-v0.1"
# model="mistralai/Mistral-7B-Instruct-v0.3"
modelname=meg-${MODELS[$model]}
echo ${model} ${modelname}

LR=1e-5
MAX_NEW_TOKENS=20
MODEL_MAX_LEN=124 # max_length = 112
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
WANDB_NAME=${DATASET}-${modelname}-lr${LR}-${MODEL_MAX_LEN}-${mapping_type}-${EMBED_DIM}-${input_prefix_length}-${output_prefix_length}-${EMBEDS_ORIGIN}-s${SEED}
logging_file=${LOGS_DIR}/logs_${DATASET}/logs_${WANDB_NAME}.txt
echo "Logging to ${logging_file}"

## ==================================================================

cd ${TRAIN_CODE_DIR}

deepspeed --include localhost:4,5,6,7 --master_port 29505 train_meg_mistral.py > ${logging_file} 2>&1 \
--deepspeed ds/deepspeed_config_s2.json \
--task ${DATASET} \
--mapping_type ${mapping_type} \
--model_name_or_path ${model} \
--modelname ${modelname} \
--data_path ${BASE_DATA_DIR}/${EMBEDS_ORIGIN}/vocab_training_meg.jsonl \
--embeddings_dir ${BASE_DATA_DIR}/${EMBEDS_ORIGIN}/embeddings/${EMBEDS_FILE} \
--output_dir ${CKPT_DIR}/${WANDB_NAME} \
--padding_side left \
--bf16 True \
--num_train_epochs 1 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
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
--xent_temperature ${loss_temp} \
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
