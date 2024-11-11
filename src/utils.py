import os
import json
import random
import torch
import h5py
import numpy as np
import pandas as pd
from typing import Any, Optional


class torch_dtype:
    def __init__(self, dtype: torch.dtype) -> None:
        self.dtype = dtype
    
    def __enter__(self) -> Any:
        self.dtype_orig = torch.get_default_dtype()
        if self.dtype is not None:
            torch.set_default_dtype(self.dtype)
    
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> Optional[bool]:
        if self.dtype is not None:
            torch.set_default_dtype(self.dtype_orig)


def set_seed(seed):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def summary_parameters(model, logger=None):
    """
    Summary Parameters of Model
    :param model: torch.nn.module_name
    :param logger: logger
    :return: None
    """

    print_and_log('>> Parameters:', logger)
    parameters = [(str(n), str(v.dtype), str(tuple(v.shape)), str(v.numel()), str(v.requires_grad))
                           for n, v in model.named_parameters()]
    max_lens = [max([len(item) + 4 for item in col]) for col in zip(*parameters)]
    raw_format = '|' + '|'.join(['{{:{}s}}'.format(max_len) for max_len in max_lens]) + '|'
    raw_split = '-' * (sum(max_lens) + len(max_lens) + 1)
    print_and_log(raw_split, logger)
    print_and_log(raw_format.format('Name', 'Dtype', 'Shape', '#Params', 'Trainable'), logger)
    print_and_log(raw_split, logger)

    for name, dtype, shape, number, grad in parameters:
        print_and_log(raw_format.format(name, dtype, shape, number, grad), logger)
        print_and_log(raw_split, logger)

    num_trainable_params = sum([v.numel() for v in model.parameters() if v.requires_grad])
    total_params = sum([v.numel() for v in model.parameters()])
    non_trainable_params = total_params - num_trainable_params
    print_and_log('>> {:25s}\t{:.2f}\tM'.format('# TrainableParams:', num_trainable_params / (1.0 * 10 ** 6)), logger)
    print_and_log('>> {:25s}\t{:.2f}\tM'.format('# NonTrainableParams:', non_trainable_params / (1.0 * 10 ** 6)), logger)
    print_and_log('>> {:25s}\t{:.2f}\tM'.format('# TotalParams:', total_params / (1.0 * 10 ** 6)), logger)


def save_model_ckpt(model, local_rank, output_dir, distributed=True):
    if distributed and local_rank != 0: # Barrier
        torch.distributed.barrier()
        
    try:
        # Save model's state dict 
        os.makedirs(f"{output_dir}/checkpoint_last", exist_ok=True)
        torch.save(
            model.state_dict(),
            f"{output_dir}/checkpoint_last/model_state_dict.pt",
        )
        if local_rank == 0:
            print(f"Ckpt at {output_dir}/checkpoint_last/model_state_dict.pt")
    except Exception as err:
        print(f"(!!) Error saving the model state dict. >>> {err}")

    if distributed and local_rank == 0: #End of barrier
        torch.distributed.barrier()


# JSONL 
def load_jsonl_file(file_path, format="list", only_letter=False):
    if file_path is None:
        return None
    if format=="dict":
        data = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                content = json.loads(line)
                key_id = list(content.keys())[0]
                if only_letter:
                    response = list(content.values())[0][-1]["content"][:2] # get only letter 
                    content[key_id][-1]["content"] = response
                data[key_id] = content[key_id]
    else:
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Parse each line as JSON and add it to the list
                data.append(json.loads(line))
    return data


def save_jsonl_file(df: pd.DataFrame, file_path: str):
    df.to_json(file_path,orient="records", lines=True)


# H5 
def load_h5_file(path, K=None, unsqueeze=True):
    data_dict = {}
    if "rdf2vec" in path:
        with h5py.File(path, "r") as f:
            for mainkey in f.keys():
                for key in f[mainkey]["pyRDF2Vec"]:
                    if unsqueeze:
                        data_dict[key.replace("___", "/").replace("__", " ")] = (
                            torch.tensor(
                                f[mainkey]["pyRDF2Vec"][key][:], dtype=torch.float16
                            ).unsqueeze(0)
                        )
                    else:
                        data_dict[key.replace("___", "/").replace("__", " ")] = (
                            torch.tensor(
                                f[mainkey]["pyRDF2Vec"][key][:], dtype=torch.float16
                            )
                        )
                    if K is not None and i == K:
                        break
    else:
        with h5py.File(path, "r") as f:
            for i, key in enumerate(f.keys()):
                if unsqueeze:
                    data_dict[key.replace("___", "/").replace("__", " ")] = (
                        torch.tensor(
                            f[key][:], dtype=torch.float16
                        ).unsqueeze(0)
                    )
                else:
                    data_dict[key.replace("___", "/").replace("__", " ")] = (
                        torch.tensor(
                            f[key][:], dtype=torch.float16
                        )
                    )
                if K is not None and i == K:
                    break
    if "p_75" in data_dict:
        data_dict['p75'] = data_dict.pop('p_75')
    print(f"{len(data_dict)} entities loaded")
    return data_dict


def save_h5_file(data, output_file_path):
    with h5py.File(output_file_path, "w") as f:
        for idx, emb in enumerate(data):
            f.create_dataset(str(idx), data=emb)
    print(f"{output_file_path} written")