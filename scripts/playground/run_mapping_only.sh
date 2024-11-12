#!/bin/bash
# set -e

export OMP_NUM_THREADS=8  # nb_cpu_threads / nproc_per_node
export PYTHONPATH=$(builtin cd ..; pwd)

export DS_SKIP_CUDA_CHECK=1

cd ../
. ../main.config
export WANDB_ENTITY=${WANDB_ENTITY}
export WANDB_PROJECT=${WANDB_PROJECT}

model="mistralai/Mistral-7B-Instruct-v0.1"
modelname=${MODELS[$model]}
echo ${model} ${modelname}

SEED=0
DATASET=umls-triplets

## Graph
EMBEDS_ORIGIN=umls
EMBEDS_FILE=word2embed_graphsage_sapbertinit.h5
EMBED_DIM=256

MODEL_MAX_LEN=64
MAX_NEW_TOKENS=50

## Mapping network
mapping_type="mlp"
input_prefix_length=1
output_prefix_length=1
num_layers=4
num_heads=1
LR=0.0002

## Logs
WANDB_NAME=${DATASET}-${mapping_type}-${num_layers}-${num_heads}-lr${LR}

cd ${TRAIN_CODE_DIR}

CUDA_VISIBLE_DEVICE=4,5 torchrun --master_port 29502 train_mapping_network.py \
--task ${DATASET} \
--mapping_type ${mapping_type} \
--model_name_or_path ${model} \
--modelname ${modelname} \
--data_path ${ROOT_DATA_DIR}/${EMBEDS_ORIGIN}/embeddings/contextualized_embeddings.h5 \
--embeddings_dir ${ROOT_DATA_DIR}/${EMBEDS_ORIGIN}/embeddings/${EMBEDS_FILE} \
--output_dir ${CKPT_DIR}/${WANDB_NAME} \
--padding_side left \
--bf16 True \
--num_train_epochs 10 \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 32 \
--save_strategy "epoch"  \
--save_steps 1 \
--logging_strategy "steps" \
--logging_steps 200 \
--save_total_limit 5 \
--learning_rate ${LR} \
--weight_decay 0.001     \
--warmup_ratio 0    \
--loss_temperature 0.5 \
--lr_scheduler_type "cosine"   \
--model_max_length ${MODEL_MAX_LEN} \
--max_new_tokens ${MAX_NEW_TOKENS} \
--gradient_checkpointing False \
--temp 0.2 \
--num_heads_map_network ${num_heads} \
--num_layers_map_network ${num_layers} \
--prefix_length ${input_prefix_length} \
--output_prefix_length ${output_prefix_length} \
--prefix_dim ${EMBED_DIM} \
--seed ${SEED} \
--run_name ${WANDB_NAME} \
--report_to "wandb"