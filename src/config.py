import torch
from mapping_networks import MappingType
from transformers import MistralConfig
from typing import List


class MEGConfig(MistralConfig):
    model_type = "MEG"
    def __init__(self, 
                 task: str="",
                 modelname: str="",
                 padding_side: str="left",
                 prefix_length: int=1, 
                 mapping_type: MappingType=MappingType.MLP, 
                 attn_implementation: str="flash_attention_2", 
                 gradient_checkpointing: bool=False,
                 torch_dtype: torch.dtype=torch.float16, 
                 use_cache: bool=False, 
                 device_map: bool=None, 
                 freeze: bool = False,
                 compound_loss: bool = False,
                 lora_r: int = 32,
                 lora_alpha: int = 64,
                 lora_dropout: float = 0.05,
                 lora_bias: str = "none",
                 lora_target_modules: List[str] = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
                 **kwargs):
        super().__init__(**kwargs)
        self.task=task
        self.modelname=modelname
        self.padding_side=padding_side
        self.prefix_length=prefix_length
        self.mapping_type=mapping_type
        self.attn_implementation=attn_implementation
        self.gradient_checkpointing=gradient_checkpointing
        self.torch_dtype=torch_dtype
        self.use_cache=use_cache
        self.device_map=device_map
        self.freeze = freeze
        self.compound_loss = compound_loss
        # LORA
        self.lora_r=lora_r
        self.lora_alpha=lora_alpha
        self.lora_dropout=lora_dropout
        self.lora_target_modules=lora_target_modules
        self.lora_bias=lora_bias


def parse_meg_config(data_args, training_args, lora_args, model_args, dtype):
    if model_args.mapping_type is None:
        mapping_type = None
    else:
        mapping_type = {
            "mlp": MappingType.MLP,
            "transformer": MappingType.Transformer,
            "linear": MappingType.Linear

        }[model_args.mapping_type]   
        
    config = MEGConfig(
            task=data_args.task,
            modelname=data_args.modelname,
            padding_side=data_args.padding_side,
            prefix_length=model_args.output_prefix_length,
            mapping_type=mapping_type,
            attn_implementation="flash_attention_2",
            gradient_checkpointing=training_args.gradient_checkpointing,
            torch_dtype=dtype,
            use_cache=True,  # set to False as we're going to use gradient checkpointing
            device_map=None,  # DO NOT change for distributed training        
            lora_r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            lora_dropout=lora_args.lora_dropout,
            lora_target_modules=lora_args.lora_target_modules,
            lora_bias=lora_args.lora_bias,
            freeze=data_args.freeze,
            compound_loss=model_args.compound_loss
    )
    return config
