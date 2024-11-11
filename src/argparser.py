import transformers
from enum import Enum
from dataclasses import dataclass, field, fields
from typing import Dict, List, Optional, Any


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    
    flash_attn: bool = False
    seed: int = 231106

    # Training
    n_gpus: int = 1
    is_rn: bool = False
    output_dir: str = field(
        default=None, metadata={"help": "Path to folder with the hinsage model."}
    )
    remove_unused_columns: bool = False
    eval_at_the_end: bool = False

    # Checkpoints and Logging
    save_every: int = 1 
    use_wandb: bool = False


@dataclass
class DataArguments:
    task: str = field(default="ersa", metadata={"help": "task or dataset name"})
    model_name_or_path: Optional[str] = field(default="google/flan-t5-small")
    modelname: Optional[str] = field(default="flan-t5-small")
    graph_in_prompt: str = None
    max_num_embeds: int = 20
    freeze: bool = False
    model_max_length: int = field(
        default=2560,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data in conversation format."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    test_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    processed_data_path: str = field(
        default=None, metadata={"help": "Path to the training data preprocessed (no conversation). Used for prefix training."}
    )
    embeddings_dir: str = field(
        default=None, metadata={"help": "Path to pickle file with dictionary of pretrained embeddings."}
    )
    concept_names: str = field(
        default=None, metadata={"help": "Path to file with mappings between umls code and entity name."}
    )
    umls_parents: str = field(
        default=None, metadata={"help": "Path to json with list of parents per entity (CUI)."}
    )
    padding_side: str = "left"
    temp: float = field(
        default=0.2, metadata={"help": "Temperature to control model's output"}
    )
    loss_temperature: float = field(
        default=0.5, metadata={"help": "Temperature to control model's output"}
    )
    # inference params
    max_new_tokens: int = 20
    top_p: float = 0.9
    top_k: int = 5


@dataclass
class LoraArguments:
    quantization_config: str = "none"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


@dataclass
class ModelArguments:
    mapping_type: str = field(default=None, metadata={"help": "mlp/transformer"})
    prefix_dim: int = 128  
    prefix_length: int = 10
    output_prefix_length: int = 1
    random_prefix: bool = False
    normalize_prefix: bool = False
    only_prefix: bool = False
    num_heads_map_network: int = 8
    num_layers_map_network: int = 8
    resume_mapping_from_checkpoint: Optional[str] = field(default="")
    compound_loss: bool = False

    def _dict_torch_dtype_to_str(self, d: Dict[str, Any]) -> None:
        """
        Checks whether the passed dictionary and its nested dicts have a *torch_dtype* key and if it's not None,
        converts torch.dtype to a string of just the type. For example, `torch.float32` get converted into *"float32"*
        string, which can then be stored in the json format.
        """
        if d.get("torch_dtype", None) is not None and not isinstance(d["torch_dtype"], str):
            d["torch_dtype"] = str(d["torch_dtype"]).split(".")[1]
        for value in d.values():
            if isinstance(value, dict):
                self._dict_torch_dtype_to_str(value)

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        # filter out fields that are defined as field(init=False)
        d = {field.name: getattr(self, field.name) for field in fields(self) if field.init}

        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"

        self._dict_torch_dtype_to_str(d)

        return d


local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)
