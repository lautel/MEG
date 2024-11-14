import os
import logging
import torch
import transformers
import torch.distributed as dist

from argparser import DataArguments, ModelArguments, LoraArguments
from config import parse_meg_config
from train_meg_llama import MEGModel
from eval_utils import main_eval, compute_eval_metrics
from utils import set_seed
from trl import SFTConfig

logger = logging.getLogger(__name__)
logging.getLogger('transformers.generation_utils').disabled = True
logger.setLevel(logging.DEBUG)

os.environ["TOKENIZERS_PARALLELISM"] = "true"


if __name__ == '__main__':
    ## Read input arguments
    parser = transformers.HfArgumentParser(
        (DataArguments, SFTConfig, LoraArguments, ModelArguments)
    )
    data_args, training_args, lora_args, model_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)

    ## Config precision
    dtype=torch.float16
    if training_args.bf16:
        dtype=torch.bfloat16
    else:
        if training_args.tf32:
            dtype=torch.float32

    ## Load MEGModel
    config = parse_meg_config(data_args, training_args, lora_args, model_args, dtype)
    if training_args.resume_from_checkpoint:
        if "global_step" in training_args.resume_from_checkpoint:
            ckpt_filepath = f"{training_args.resume_from_checkpoint}/mp_rank_00_model_states.pt"
        elif "checkpoint-" in training_args.resume_from_checkpoint:
            ckpt_filepath = f"{training_args.resume_from_checkpoint}/pytorch_model.bin"
        else:
            ckpt_filepath = f"{training_args.resume_from_checkpoint}/model_state_dict.pt"
        model =  MEGModel.from_pretrained(
            checkpoint_path=ckpt_filepath, 
            dtype=dtype,
            config=config,
            data_args=data_args, 
            model_args=model_args
        )
    else:
        model =  MEGModel(config, data_args, model_args)

    ## Evaluate
    main_eval(model, dtype, data_args, training_args, model_args)

    if dist.get_rank() == 0 and "umls" not in data_args.task:
        compute_eval_metrics(data_args.task, training_args.output_dir, training_args.seed)
