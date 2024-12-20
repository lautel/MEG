import logging
import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          PreTrainedModel)
from peft import LoraConfig, get_peft_model
from pathlib import Path
from typing import Optional, Union, List, Callable, Any
from config import MEGConfig
from mapping_networks import LinearMappingNetwork, MultiHeadMLPMappingNetwork, TransformerMappingNetwork, MappingType
# from huggingface_hub import login
# login(token="your-hugginface-login-token")

logger = logging.getLogger(__name__)
logging.getLogger("transformers.generation_utils").disabled = True
logger.setLevel(logging.DEBUG)

os.environ["TOKENIZERS_PARALLELISM"] = "true"


MEG_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "lautel/MEG-Mistral-7B-Instruct-v0.1",
    "lautel/MEG-Mistral-7B-Instruct-v0.3",
    "lautel/MEG-Llama-3.1-8B-Instruct"
]


class MEGPreTrainedModel(PreTrainedModel):
    """
    Abstract class.
    """

    config_class = MEGConfig
    base_model_prefix = "meg-instruct"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing_enable()
        self.enable_input_require_grads()
        if isinstance(module, AutoModelForCausalLM):
            module.gradient_checkpointing = value
            module.gradient_checkpointing_enable()
            module.enable_input_require_grads()


class MEGModel(MEGPreTrainedModel):
    def __init__(
        self,
        config,
        data_args,
        model_args
        ) -> None:
        super().__init__(config)
        self.config = config

        try:
            self.local_rank = os.environ["LOCAL_RANK"]
        except KeyError:
            self.local_rank = 0

        # 0. Define placeholder for constants
        self.pad_token = None
        self.response_template = None
        self.kg_embedding_pattern = None

        # 1. Graph
        # Done offline 

        # 2. Mapping Network
        output_dim=4096  # self.language_decoder.embed_dim,
        hidden_dim=128
        if self.config.mapping_type == MappingType.Linear:
            self.mapper = LinearMappingNetwork(
                input_dim=model_args.prefix_dim,
                output_dim=output_dim,
                dtype=self.config.torch_dtype,
            )
        elif self.config.mapping_type == MappingType.Transformer:
            self.mapper = TransformerMappingNetwork(
                output_dim=output_dim,
                hidden_dim=hidden_dim,
                t=data_args.xent_temperature,
                model_args=model_args,
                dtype=self.config.torch_dtype
            )
            if model_args.resume_mapping_from_checkpoint:
                self.mapper.from_pretrained(model_args.resume_mapping_from_checkpoint)
                # Freeze the mapping function -- weights have been pretrained offline
                for param in self.mapper.parameters():
                    param.requires_grad = False
        elif self.config.mapping_type == MappingType.MLP:
            self.mapper = MultiHeadMLPMappingNetwork(
                output_dim=output_dim,
                hidden_dim=hidden_dim,
                t=data_args.xent_temperature,
                model_args=model_args,
                dtype=self.config.torch_dtype
            )
            if model_args.resume_mapping_from_checkpoint:
                self.mapper.from_pretrained(model_args.resume_mapping_from_checkpoint)
                # Freeze the mapping function -- weights have been pretrained offline
                for param in self.mapper.parameters():
                    param.requires_grad = False
        else:
            self.mapper = None
        self.freeze_lm = self.config.freeze

    
        # 3. LLM
        if data_args.temp < 1e-4:
            self.do_sample = False
        else:
            self.do_sample = True

        self.language_decoder = AutoModelForCausalLM.from_pretrained(
                data_args.model_name_or_path,
                attn_implementation=self.config.attn_implementation,
                torch_dtype=self.config.torch_dtype,
                temperature=data_args.temp,
                do_sample=self.do_sample,
                use_cache=self.config.use_cache,  
                device_map=self.config.device_map, 
            )

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias=self.config.lora_bias,
            task_type="CAUSAL_LM",
        )
        self.language_decoder = get_peft_model(self.language_decoder, lora_config)
        if self.config.gradient_checkpointing:
            self._set_gradient_checkpointing(self.language_decoder)

        if self.freeze_lm:
            for param in self.language_decoder.parameters():
                param.requires_grad = False

        self.padding_side = config.padding_side
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.text_processor = self._initialize_tokenizer_and_embeds(
            data_args.model_name_or_path, 
            data_args.model_max_length)
        self.embed_tokens = self.get_input_embeddings()

    def set_tokenizer(self, value):
        self.text_processor = value
    
    def get_tokenizer(self):
        return self.text_processor

    def _initialize_tokenizer_and_embeds(self, model_name_or_path, model_max_length):
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=None,
            padding=True,
            truncation=True,
            padding_side=self.padding_side,
            truncation_side=self.padding_side,
            use_fast=False,
        ) 
        # Set reasonable default for models without max length
        if tokenizer.model_max_length > 100_000:
            tokenizer.model_max_length = model_max_length

        if self.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = self.pad_token

        self.language_decoder.orig_embeds_params = [self.get_input_embeddings().weight.data.clone()]
        for p in self.get_input_embeddings().parameters():
            p.requires_grad = True
        for p in self.get_output_embeddings().parameters():
            p.requires_grad = False
        return tokenizer

    def get_input_embeddings(self) -> nn.Module:
        return self.language_decoder.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_decoder.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_decoder.get_output_embeddings()

    def resize_token_embeddings(self, size: int) -> nn.Module:
        return self.language_decoder.resize_token_embeddings(size)
    
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Any,
        use_cache: bool,
        output_attentions: bool,
    ):
        self.language_decoder._update_causal_mask(
            attention_mask,
            input_tensor,
            cache_position,
            past_key_values,
            use_cache,
            output_attentions,
        )
    
    def save_pretrained(self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "5GB",
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        save_peft_format: bool = True,
        **kwargs,
    ):
        print(">>> save pretrained")
        sys.stdout.flush()

        if self.local_rank != 0: #Start barrier
            torch.distributed.barrier()

        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return
             
        try:
            os.makedirs(f"{save_directory}", exist_ok=True)
            # Save model's state dict 
            torch.save(
                state_dict,
                f"{save_directory}/model_state_dict.pt",
            )
            logger.info(f"Ckpt at {save_directory}/model_state_dict.pt")
        except Exception as err:
            print(f"(!!) Error saving the model state dict. >>> {err}")

        if self.local_rank == 0: #End of barrier
            torch.distributed.barrier()
    
    def mapping_from_pretrained(self, checkpoint_path: Union[str, Path]):
        self.mapper.from_pretrained(checkpoint_path)
    
    def embed_graph(self, graph_embeds: torch.Tensor, target_embeds=None) -> torch.Tensor:
        if self.config.torch_dtype != graph_embeds.dtype:
            graph_embeds = graph_embeds.type(self.config.torch_dtype)
        results = self.mapper(graph_embeds, target_embeds)
        return results["logits"], results["loss"]
    
    def embed_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            token_embeds = self.embed_tokens(input_ids)
        return token_embeds

    def _get_input_embeds(self, input_ids, context_embeds, graph_embeds):
        # Compose input for LLM
        # input_embeds = torch.cat((graph_embeds, context_embeds), dim=1)  # base setup
        # find position of {kg_embedding} in text
        start_idxs, template_size = self.find_subarray_positions(input_ids, self.kg_embedding_pattern)
        assert len(start_idxs) == input_ids.shape[0]
        input_embeds_out, input_ids_out = [], []
        for i, start_idx in enumerate(start_idxs):
            if start_idx == -1:
                input_embeds_out.append(context_embeds[i])
                input_ids_out.append(input_ids[i])
                continue
            _input_embeds = torch.cat((context_embeds[i][:start_idx, :], 
                                    graph_embeds[i], 
                                    context_embeds[i][start_idx + template_size:, :]), dim=0)
            # Adjust input_ids to correctly compute attention_mask later
            placeholder = -1 * torch.ones((graph_embeds.shape[1]))
            _input_ids = torch.cat((input_ids[i][:start_idx], 
                            placeholder.to(dtype=input_ids.dtype, device=input_ids.device), 
                            input_ids[i][start_idx + template_size:]), dim=0)
            input_embeds_out.append(_input_embeds)
            input_ids_out.append(_input_ids)
        if len(input_ids_out) > 1:
            # ids
            max_len = max([t.shape[0] for t in input_ids_out])
            input_ids_out = [
                F.pad(t, (max_len - t.shape[0], 0), value=self.text_processor.pad_token_id)
                for t in input_ids_out
            ]
            # embeds
            max_len = max([t.shape[0] for t in input_embeds_out])
            input_embeds_out = [
                F.pad(t, (0, 0, max_len - t.shape[0], 0), value=0)
                if t.shape[0] != max_len else t for t in input_embeds_out 
            ]
        input_ids_out = torch.stack(input_ids_out)
        input_embeds_out = torch.stack(input_embeds_out)

        return input_embeds_out, input_ids_out
    
    # This function will return the first position of the subarray if it exists
    @staticmethod
    def find_subarray_positions(input_ids, pattern):
        assert len(input_ids.size()) == 2
        batch_size, seq_length = input_ids.size()
        template_size = pattern.size(0)
        pattern = pattern.to(input_ids.device)
        positions = []
        for i in range(batch_size):
            found_position = -1
            for j in range(seq_length - template_size + 1):
                if torch.equal(input_ids[i, j:j + template_size], pattern):
                    found_position = j
                    break
            positions.append(found_position)
        return positions, template_size

    def do_forward_pass(self, input_ids, prefix, labels=None, is_training=True):

        input_embeds = self.embed_text(input_ids)

        target_embeds = None
        if is_training and self.config.compound_loss:
            # Extract target_embeds to compute compound loss in the mapping network. 
            start_label_pos, template_size_st = self.find_subarray_positions(input_ids, self.response_template[self.config.task][self.config.modelname])
            end_label_pos, _ = self.find_subarray_positions(input_ids, torch.tensor([2]))
            target_embeds = []
            for i, (st, end) in enumerate(zip(start_label_pos, end_label_pos)):
                st += template_size_st
                target_embeds_i = input_embeds[i, st:end, :]
                # Mean pool over the embeddings of the entity's tokens
                target_embeds.append(torch.mean(target_embeds_i, dim=0).unsqueeze(0))

            target_embeds = torch.stack( target_embeds )

        if self.mapper is None:
            graph_embeds, loss = None, None
        else:
            graph_embeds, loss = self.embed_graph(prefix, target_embeds)
        attention_mask = None
        new_labels = []

        if "umls" in self.config.task:  # Train mapping between embedding spaces
            # THIS TASK ALWAYS HAS PREFIX AND GRAPH_EMBEDS, SO NO NEED TO ADD CONTROL FOR THIS
            # Right before [/INST]
            positions, template_size = self.find_subarray_positions(input_ids, self.response_template[self.config.task][self.config.modelname])
            # Insert tensor A at the located positions in input_embeds 
            # i.e. replacing the input_embed at the position right before [/INST],
            # which was a "double space" placeholder (id=28705 in mistral-instruct-v0.1)
            new_input_ids, new_input_embeds = [], []
            for i, pos in enumerate(positions):
                if pos != -1:  # Ensure the template was found
                    # input_embeds[i, pos:pos + template_size, :] = graph_embeds[i,:]
                    left_part = input_embeds[i, :pos, :]  # Sequence up to `pos`
                    right_part = input_embeds[i, pos + template_size:, :]  # Sequence after `pos`
                    new_input_embeds.append( torch.cat((left_part, graph_embeds[i], right_part), dim=0) )

                    ids_left_part = input_ids[i, :pos]  # Sequence up to `pos`
                    ids_right_part = input_ids[i, pos + template_size:]  # Sequence after `pos`
                    placeholder = -1 * torch.ones((graph_embeds.shape[1]))
                    new_input_ids.append( torch.cat((ids_left_part, 
                                                     placeholder.to(dtype=input_ids.dtype, device=input_ids.device), 
                                                     ids_right_part), 
                                                    dim=0) )
                    if labels is not None:
                        ids_left_part = labels[i, :pos]  # Sequence up to `pos`
                        ids_right_part = labels[i, pos + template_size:]  # Sequence after `pos`
                        placeholder = -100 * torch.ones((graph_embeds.shape[1]))
                        new_labels.append( torch.cat((ids_left_part, 
                                                        placeholder.to(dtype=input_ids.dtype, device=input_ids.device), 
                                                        ids_right_part), 
                                                        dim=0) )
            if len(new_input_ids) > 1:
                # ids
                max_len = max([t.shape[0] for t in new_input_ids])
                new_input_ids = [
                    F.pad(t, (max_len - t.shape[0], 0), value=self.text_processor.pad_token_id)
                    for t in new_input_ids
                ]
                # labels
                if new_labels:
                    new_labels = [
                        F.pad(t, (max_len - t.shape[0], 0), value=self.text_processor.pad_token_id)
                        for t in new_labels
                    ]
                # embeds
                new_input_embeds = [
                    F.pad(t, (0, 0, max_len - t.shape[0], 0), value=0)
                    if t.shape[0] != max_len else t for t in new_input_embeds 
                ]

            new_input_ids = torch.stack(new_input_ids)
            new_input_embeds = torch.stack(new_input_embeds)
            if new_labels:
                new_labels = torch.stack(new_labels)
        else:
            if graph_embeds is not None:
                new_input_embeds, new_input_ids = self._get_input_embeds(input_ids, input_embeds, graph_embeds)
                # -100 = ignore_index when computing loss
                if labels is not None:
                    if len(new_input_ids.size()) == 1:
                        size_increase = new_input_ids.shape[0] - labels.shape[1]
                    else:
                        size_increase = new_input_ids.shape[1] - labels.shape[1]
                    if size_increase >= 0:
                        new_labels = torch.cat((
                            -100*torch.ones((labels.shape[0], size_increase), dtype=torch.int, device=labels.device), labels), 
                        dim=1)
                    else:
                        new_labels = labels[:,-1*size_increase:]
            else:
                new_input_embeds, new_input_ids, new_labels = input_embeds, input_ids, labels

        attention_mask = (new_input_ids != self.text_processor.pad_token_id).long()
        return new_input_embeds, attention_mask, new_labels, loss  

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        prefix: torch.Tensor = None,
        qid: torch.Tensor = None,
        label_ids: torch.Tensor = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> torch.Tensor:
        if self.config.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

        input_embeds, attention_mask, new_labels, mapping_loss = self.do_forward_pass(input_ids, prefix, labels, is_training=True)
        # print(print(torch.mul(input_ids, attention_mask[:,:input_ids.shape[1]])))
        # print(self.text_processor.decode( new_labels[new_labels==-100] = 0  ))

        if self.config.gradient_checkpointing and self.training:
            outputs = self._gradient_checkpointing_func(
                self.language_decoder.__call__,
                input_embeds,
                attention_mask,
                new_labels,
                use_cache
            )
        else:
            outputs = self.language_decoder(inputs_embeds=input_embeds, attention_mask=attention_mask, labels=new_labels, use_cache=use_cache)
            """
            CausalLMOutputWithPast(loss=tensor(0.3545, device='cuda:0', grad_fn=<NllLossBackward0>), 
                                   logits=tensor([[[ ...]]], device='cuda:0', grad_fn=<ToCopyBackward0>), 
                                   past_key_values=None, hidden_states=None, attentions=None)
            """
        if mapping_loss is not None:
            outputs["loss"] += mapping_loss

        return outputs

    
    @torch.inference_mode()
    def generate(
        self,
        task: str,
        prefix: torch.Tensor,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        output_scores: bool = False,
        return_dict_in_generate: bool=False,
        debug: bool=True,
        **kwargs
    ) -> List[str]:
        
        input_embeds, attention_mask, _, _ = self.do_forward_pass(input_ids, prefix, is_training=False)
        
        outputs = self.language_decoder.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            eos_token_id=self.text_processor.eos_token_id,
            pad_token_id=self.text_processor.pad_token_id,
            max_new_tokens=max_new_tokens,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate)
        
        if debug:
            transition_scores = self.language_decoder.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True)
            generated_tokens = outputs.sequences
            for seq_id in range(len(generated_tokens)):
                for tok, score in zip(generated_tokens[seq_id], transition_scores[seq_id]):
                    # | token | token string | log probability | probability
                    print(f"| {tok:5d} | {self.text_processor.decode(tok):8s} | {score.cpu().numpy():.3f} | {np.exp(score.cpu().numpy()):.2%}\n")

            import pdb; pdb.set_trace()

        return outputs
    
    def text_transform(self, text: Union[str, List[str]], **kwargs) -> torch.Tensor:
        return self.text_processor(text, padding='longest', return_tensors='pt', **kwargs)
    

class MEGLastLayerModel(MEGModel):
    def __init__(
        self,
        config,
        data_args,
        model_args
        ) -> None:
        super().__init__(config, data_args, model_args)

        self.graph_embeds, self.inject_position, self.template_size = None, None, None


    def set_injection(self, input_ids, prefix):
        """
        Sets up the embeddings and position for injection during generation.
        """
        self.graph_embeds = self.embed_graph(prefix)
        self.inject_position, self.template_size = self.find_subarray_positions(input_ids, self.response_template[self.config.task][self.config.modelname])
        if "umls" in self.config.task:
            self.template_size = 0  # There's no template that needs to be removed


    def inject_embeds_forward_pass(self, module, inputs, outputs):
        if type(outputs) is torch.Tensor:
            last_hidden_state = outputs
        else:
            last_hidden_state = outputs.hidden_states[-1]  # [batch_size, seq_length, hidden_size]
        
        # Ensure graph_embeds has the correct shape and inject it
        if self.graph_embeds.shape[-1] != self.language_decoder.embed_dim:
            raise ValueError(f"graph_embeds last dimension must be {self.language_decoder.embed_dim} but got {self.graph_embeds.shape[-1]}")
        
        # Insert tensor A at the located positions in input_embeds 
        # i.e. replacing the input_embed at the position right before [/INST], 
        # For UMLS is a "double space" placeholder (id=28705). For MedQA is the string {kg_embedding}
        combined_hidden_state = []
        for i, pos in enumerate(self.inject_position):
            if pos != -1:  # Ensure the template was found
                # last_hidden_state[i, pos:pos + template_size, :] = graph_embeds[i,:]
                left_part = last_hidden_state[i, :pos, :]  # Sequence up to `pos`
                right_part = last_hidden_state[i, pos + self.template_size:, :]  # Sequence after `pos`
                combined_hidden_state.append( torch.cat((left_part, self.graph_embeds[i], right_part), dim=0) )
            else:
                combined_hidden_state.append( last_hidden_state[i] )

        if len(combined_hidden_state) > 1:
            max_len = max([t.shape[0] for t in combined_hidden_state])
            combined_hidden_state = [
                F.pad(t, (0, 0, max_len - t.shape[0], 0), value=0)
                if t.shape[0] != max_len else t for t in combined_hidden_state 
            ]
            
        combined_hidden_state = torch.stack(combined_hidden_state)
        
        # Pass combined embeddings through the final layer to get the logits
        logits = self.language_decoder.lm_head(combined_hidden_state)
        # Re-assign logits
        outputs.logits = logits
        return outputs

    def forward(
        self,
        prefix: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        qid: torch.Tensor = None,
        label_ids: torch.Tensor = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> torch.Tensor:
        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

        input_embeds = self.embed_text(input_ids)
        
        if self.gradient_checkpointing and self.training:
            pre_outputs = self._gradient_checkpointing_func(
                self.language_decoder.__call__,
                input_embeds,
                attention_mask,
                labels,
                use_cache,
                output_hidden_states=True
            )
        else:
            pre_outputs = self.language_decoder(inputs_embeds=input_embeds, 
                                            attention_mask=attention_mask,
                                            labels=labels,
                                            use_cache=use_cache,
                                            output_hidden_states=True
                                            )
            """
            CausalLMOutputWithPast(loss=tensor(0.3545, device='cuda:0', grad_fn=<NllLossBackward0>), 
                                   logits=tensor([[[ ...]]], device='cuda:0', grad_fn=<ToCopyBackward0>), 
                                   past_key_values=None, hidden_states=tensor([[[ ...]]], attentions=None)
            """

        self.set_injection(input_ids, prefix)
        outputs = self.inject_embeds_forward_pass(None, input_ids, pre_outputs)

        return outputs

    @torch.inference_mode()
    def generate(
        self,
        task: str,
        prefix: torch.Tensor,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        output_scores: bool = False,
        return_dict_in_generate: bool=False,
        debug: bool=True,
        **kwargs
    ) -> List[str]:
        
        input_embeds = self.embed_text(input_ids)
        attention_mask = (input_ids != self.text_processor.pad_token_id).long()

        # Register hook to inject embeddings
        self.set_injection(input_ids, prefix)
        handle = self.language_decoder.model.model.norm.register_forward_hook(self.inject_embeds_forward_pass)
        # outputs = self.inject_embeds_forward_pass(None, input_ids, pre_outputs)

        outputs = self.language_decoder.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            eos_token_id=self.text_processor.eos_token_id,
            pad_token_id=self.text_processor.pad_token_id,
            max_new_tokens=max_new_tokens,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
            **kwargs)

        # Remove the hook after generation to restore normal functionality
        handle.remove()

        return outputs
    

class MEGMistralModel(MEGModel):
    def __init__(
        self,
        config,
        data_args,
        model_args
        ) -> None:
        super().__init__(config, data_args, model_args)
        
        # Assign Mistral-based constants
        self.response_template = {
            "umls": {
                "meg-mistral-instruct-v0.1": torch.tensor([733, 28748, 16289, 28793]),
                "meg-mistral-instruct-v0.3": torch.tensor([4])
            },  # '  [/INST]'
            "umls-extended": {
                "meg-mistral-instruct-v0.1": torch.tensor([28705, 733, 28748, 16289, 28793])
            },  # '  [/INST]'
        }
        self.kg_embedding_pattern = torch.tensor([371, 8087, 28730, 18320, 3202, 28752])  # {kg_embedding}
        

class MEGLlamaModel(MEGModel):
    def __init__(
        self,
        config,
        data_args,
        model_args
        ) -> None:
        super().__init__(config, data_args, model_args)
        
        # Assign Llama-based constants
        self.pad_token = "<|finetune_right_pad_id|>"
        self.response_template = {
            "umls": {
                "meg-llama3-instruct-8B": torch.tensor([128007]),  # end instruction
                "llama3-instruct-8B": torch.tensor([128007])  # end instruction
            }
        }
        self.kg_embedding_pattern = torch.tensor([314, 7501, 52602, 92])  # {kg_embedding}


class MEGLastLayerMistralModel(MEGLastLayerModel):
    def __init__(
        self,
        config,
        data_args,
        model_args
        ) -> None:
        super().__init__(config, data_args, model_args)
        
        # Assign Mistral-based constants
        self.response_template = {
            "umls": {
                "meg-mistral-instruct-v0.1": torch.tensor([733, 28748, 16289, 28793]),
                "meg-mistral-instruct-v0.3": torch.tensor([4])
            },  # '  [/INST]'
            "umls-extended": {
                "meg-mistral-instruct-v0.1": torch.tensor([28705, 733, 28748, 16289, 28793])
            },  # '  [/INST]'
        }
        self.kg_embedding_pattern = torch.tensor([371, 8087, 28730, 18320, 3202, 28752])  # {kg_embedding}
        