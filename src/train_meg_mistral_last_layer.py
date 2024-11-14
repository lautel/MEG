import torch
import transformers 

from trl import SFTConfig
from argparser import DataArguments, LoraArguments, ModelArguments
from data_collators import  DataCollatorForCompletionOnlyLM
from utils import summary_parameters, save_model_ckpt, set_seed
from config import parse_meg_config
from meg.modeling_meg import MEGLastLayerMistralModel
from meg.trainer import MEGTrainer
from meg.data.custom_datasets import load_custom_dataset


def main():
    ## Read input arguments
    parser = transformers.HfArgumentParser(
        (DataArguments, SFTConfig, LoraArguments, ModelArguments)
    )
    data_args, training_args, lora_args, model_args = (
        parser.parse_args_into_dataclasses()
    )

    ## Set config
    set_seed(training_args.seed)
    dtype=torch.float16
    if training_args.bf16:
        dtype=torch.bfloat16
    else:
        if training_args.tf32:
            dtype=torch.float32

    ## Load MEGLastLayerMistralModel
    config = parse_meg_config(data_args, training_args, lora_args, model_args, dtype)
    model =  MEGLastLayerMistralModel(config, data_args, model_args)
    tokenizer = model.get_tokenizer()

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
    if training_args.local_rank == 0:
        summary_parameters(model)

    ## Load datasets & Data collator
    train_dataset = load_custom_dataset(data_args, model_args, tokenizer, dtype, True)
      
    # https://huggingface.co/docs/transformers/en/main_classes/data_collator#transformers.DataCollatorForLanguageModeling
    response_template = {
        "meg-mistral-instruct-v0.1": [28748, 16289, 28793],
        "meg-mistral-instruct-v0.3": [4],
        "mistral-instruct-v0.1": [28748, 16289, 28793],
        "mistral-instruct-v0.3": [4]
    }
    collator = DataCollatorForCompletionOnlyLM(
        response_template[config.modelname],
        tokenizer=tokenizer)

    ## Initialize trainer
    training_args.max_seq_length=data_args.model_max_length
    trainer = MEGTrainer(
        model=model, 
        tokenizer=tokenizer, 
        train_dataset=train_dataset,
        args=training_args,
        data_collator=collator,
        task=data_args.task
    )

    ## Train
    trainer.train()
    # TODO. Fix this to make it compatible w/ continue pretraining 
    # if training_args.resume_from_checkpoint:
    #     trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    # else:
    #     trainer.train()

    # SAVE MODEL
    save_model_ckpt(model, training_args.local_rank, training_args.output_dir)
   
if __name__ == "__main__":
    main()
