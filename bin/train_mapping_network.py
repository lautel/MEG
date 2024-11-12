import os
import sys
import logging
import random
import wandb
import numpy as np

import torch
from torch.utils.data import DataLoader

import transformers
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from trl import SFTTrainer, SFTConfig
from argparser import DataArguments, ModelArguments
from custom_datasets import load_custom_dataset
from utils import summary_parameters, torch_dtype, save_model_ckpt, set_seed
from train_utils import UniqueSampler, MinTwoLabelBatchSampler
from mapping_networks import MappingType, TransformerMappingNetwork

# from huggingface_hub import login

logger = logging.getLogger(__name__)
logging.getLogger("transformers.generation_utils").disabled = True
logger.setLevel(logging.DEBUG)

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def main():
    parser = transformers.HfArgumentParser((DataArguments, SFTConfig, ModelArguments))
    data_args, training_args, model_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

    def train_transformer(data_args, training_args, model_args):
        dtype = torch.float16
        if training_args.bf16:
            dtype = torch.bfloat16
        else:
            if training_args.tf32:
                dtype = torch.float32
        if "wandb" in training_args.report_to:
            wandb.init(
                project="MEG",
                entity="your-entity-name",
                name=training_args.run_name,
                config=training_args.__dict__,
            )
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        ####################
        ### LM Tokenizer ###
        ####################
        tokenizer = AutoTokenizer.from_pretrained(
            data_args.model_name_or_path,
            cache_dir=None,
            padding=True,
            truncation=True,
            # padding_side=self.padding_side,
            # truncation_side=self.padding_side,
            use_fast=False,
        )
        tokenizer.pad_token = tokenizer.unk_token
        if "biomistral" in data_args.model_name_or_path:
            tokenizer.pad_token = tokenizer.eos_token

        ####################
        ### Load dataset ###
        ####################
        train_dataset = load_custom_dataset(
            data_args, model_args, tokenizer, dtype, True
        )
        # data_size = len(dataset)
        # train_dataset, dev_dataset = torch.utils.data.random_split(dataset, [int(0.9*data_size), data_size-int(0.9*data_size)])
        # print(f"Training set size: {len(train_dataset)}")
        # print(f"Validation set size: {len(dev_dataset)}")

        #######################
        ### Mapping Network ###
        #######################
        input_dim = 256
        hidden_dim = 128
        output_dim = 4096  # self.language_decoder.embed_dim,
        assert model_args.mapping_type == MappingType.Transformer.value
        mapper = TransformerMappingNetwork(
            output_dim=output_dim,  # self.language_decoder.embed_dim,
            hidden_dim=hidden_dim,
            t=data_args.loss_temperature,
            model_args=model_args,
            dtype=dtype,
        )

        # if training_args.local_rank == 0:
        #     summary_parameters(mapper)

        #####################
        ### TRAINING LOOP ###
        #####################
        mapper.to(device)
        mapper.train()

        # train_sampler = UniqueSampler(train_dataset, batch_size=training_args.per_device_train_batch_size)
        train_sampler = MinTwoLabelBatchSampler(
            train_dataset, batch_size=training_args.per_device_train_batch_size
        )
        train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler)

        optimizer = torch.optim.AdamW(
            mapper.parameters(), lr=training_args.learning_rate
        )
        steps_in_one_epoch = len(train_dataloader)
        batches_in_one_epoch = (
            steps_in_one_epoch // training_args.per_device_train_batch_size
        )
        num_training_steps = int(training_args.num_train_epochs * steps_in_one_epoch)
        num_warmup_steps = int(training_args.warmup_ratio * num_training_steps)
        assert training_args.lr_scheduler_type == "linear"
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        print(f"{num_training_steps} training steps, {num_warmup_steps} warmup")
        # progress_bar = tqdm(range(num_training_steps))
        global_step = 0
        for epoch in range(int(training_args.num_train_epochs)):
            for batch in train_dataloader:
                codes, kg_embed, lm_embed_centroid = (
                    batch["input_codes"],
                    batch["input_embeds"],
                    batch["labels"],
                )
                global_step += training_args.per_device_train_batch_size
                kg_embed = kg_embed.to(device, dtype=dtype)
                lm_embed_centroid = lm_embed_centroid.to(device, dtype=dtype)
                output = mapper(kg_embed, lm_embed_centroid, codes=codes)
                loss = output["loss"]
                xent_loss = output["xent_loss"]
                back_trans_loss = output["back_trans_loss"]
                # updates
                # epoch_loss += loss
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                # progress_bar.update(1)

                if global_step % 500 == 0:
                    print(
                        f"epoch {epoch} | train loss: {loss} | contrastive_loss {xent_loss}, back_trans_loss {back_trans_loss}"
                    )
                    if "wandb" in training_args.report_to:
                        wandb.log(
                            {
                                "loss": loss,
                                "contrastive_loss": xent_loss,
                                "back_trans_loss": back_trans_loss,
                                "epoch": epoch,
                                "step": global_step,
                            }
                        )
                    sys.stdout.flush()

            # # Eval
            # mapper.eval()
            # val_epoch_loss = 0
            # for batch in dev_dataloader:
            #     codes, kg_embed, lm_embed = batch["input_codes"], batch["input_embeds"], batch["labels"]
            #     kg_embed = kg_embed.to(device)
            #     lm_embed = lm_embed.to(device)
            #     with torch.no_grad():
            #         output = mapper(codes, kg_embed, lm_embed)
            #         loss = output["loss"]
            #     # updates
            #     val_epoch_loss += loss
            # print(f">>> epoch {epoch} | train loss: {epoch_loss} | eval loss: {val_epoch_loss}")
            # if training_args.report_to == "wandb":
            #     wandb.log(
            #             {
            #                 "loss": epoch_loss,
            #                 "val_loss": val_epoch_loss,
            #                 "step": step,
            #             }
            #         )

            # SAVE MODEL
            output_dir = f"{training_args.output_dir}-t{data_args.loss_temperature}-lr{training_args.learning_rate}-s{training_args.seed}"
            save_model_ckpt(
                mapper, training_args.local_rank, output_dir, distributed=False
            )

    train_transformer(data_args, training_args, model_args)


if __name__ == "__main__":
    main()
