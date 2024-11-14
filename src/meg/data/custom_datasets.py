import os
import sys
import csv
import random
import pickle
import h5py
import logging
import pandas as pd

from typing import Tuple, List
from utils import load_jsonl_file, load_h5_file
from tqdm.auto import tqdm 

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


logger = logging.getLogger(__name__)
logging.getLogger("transformers.generation_utils").disabled = True
logger.setLevel(logging.DEBUG)


class MedQADataset(Dataset):
    def __init__(self, args, dtype, tokenizer, normalize_prefix, prefix_length, random_prefix, is_training, is_inference=False):
        self.modelname = args.modelname
        self.normalize_prefix = normalize_prefix
        self.prefix_length = prefix_length
        self.random_prefix = random_prefix
        self.is_training = is_training
        self.is_inference = is_inference
        self.graph_in_prompt = args.graph_in_prompt
        self.embeddings_dir = args.embeddings_dir
        self.umls_parents =  load_jsonl_file(args.umls_parents, "dict")
        self.tokenizer = tokenizer
        self.max_num_embeds = 20
        
        # FOR MISTRAL-INSTRUCT MODELS
        self.instruction_template = {
            "meg-mistral-instruct-v0.1": [733, 16289, 28793],
            "meg-mistral-instruct-v0.3": [3],
            "mistral-instruct-v0.1": [733, 16289, 28793],
            "mistral-instruct-v0.3": [3]
        }
    
        self.label2id = {"A)": 0, "B)": 1, "C)": 2,  "D)": 3}
        self.id2label = {v: k for k, v in self.label2id.items()}

        if is_training:
            self.split = "train"
            data_path = args.data_path
        else:
            if is_inference: 
                self.split = "test"
                data_path = args.test_data_path
            else: 
                self.split = "dev"
                data_path = args.eval_data_path

        (
            self.user_prompts,
            self.question_ids,
            self.input_ids,
            self.attn_masks,
            self.labels,
            self.label_ids,
            self.grounded 
        ) = self.load_data(data_path)

        self.prefixes = None
        
        if self.embeddings_dir is not None:
            self.prefixes = self.load_embeddings(dtype, args.embeddings_dir, max_num_embeds=self.max_num_embeds)

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, ...]:
        prefix = []
        if self.prefixes is not None:
            if self.random_prefix:
                # rnd_id = random.choice(list(self.input_ids.keys()))
                # prefix = self.prefixes[rnd_id]  # (prefix_dim)
                prefix = torch.rand(self.prefixes[i].shape, dtype=torch.float16)
            else:
                prefix = self.prefixes[ i ]  # (prefix_dim)
            if self.normalize_prefix:
                prefix = prefix.float()
                prefix = prefix / prefix.norm(2, -1)

        if "meg" in self.modelname:
            output = dict(
                qid=i,
                input_ids=self.input_ids[i],
                attention_mask=self.attn_masks[i],
                labels=self.labels[i],
                label_ids=self.label_ids[i],
                prefix=prefix
            )  
        else:
            if self.is_inference:
                output = dict(
                    input_ids=self.input_ids[i],
                    label_ids=self.label_ids[i]
                )
            else:
                output = dict(
                    input_ids=self.input_ids[i],
                    attention_mask=self.attn_masks[i],
                    labels=self.labels[i]
                )
        return output

    def tokenize_and_apply_template(self, conv):
        """
        conv=[{'role': 'user',
               'content': "Please, address ..."},
              {'role': 'assistant',
               'content': "A) Negative for the patient's health"}]
        """
    
        if not self.is_inference:
            _user_prompt = conv[0]["content"]
            tokenizer_out = self.tokenizer.apply_chat_template(
                conv, return_tensors="pt", padding="max_length", return_dict=True
            )
        else:
            conv_formatted = self.tokenizer.apply_chat_template(conv, tokenize=False)
            # Remove the initial <s> ([3:]) as it will be added by the tokenizer again later
            _user_prompt = conv_formatted.split("[/INST]")[0][4:] + "[/INST]"
            tokenizer_out = self.tokenizer(
                _user_prompt, return_tensors="pt", padding="max_length", return_attention_mask=True
            )
        _input_ids = tokenizer_out.input_ids.squeeze(0)
        _attn_masks = tokenizer_out.attention_mask.squeeze(0)
        # Truncate if necessary
        bs = _input_ids.shape[0]
        if _input_ids.shape[-1] > self.tokenizer.model_max_length:
            if self.tokenizer.truncation_side == "right":
                # 1. truncate
                _input_ids = _input_ids[: self.tokenizer.model_max_length]
                _attn_masks = _attn_masks[: self.tokenizer.model_max_length]
            else:
                # 1. truncate
                _input_ids = _input_ids[
                    -(
                        self.tokenizer.model_max_length
                        - len(self.instruction_template[self.modelname])
                    ) :
                ]
                _attn_masks = _attn_masks[
                    -(
                        self.tokenizer.model_max_length
                        - len(self.instruction_template[self.modelname])
                    ) :
                ]
                # 2. add instruction
                _input_ids = torch.cat(
                    (torch.tensor(self.instruction_template[self.modelname]), _input_ids)
                )
                _attn_masks = torch.cat(
                    (
                        torch.ones(len(self.instruction_template[self.modelname]), dtype=torch.int),
                        _attn_masks,
                    )
                )
        return _user_prompt, _input_ids, _attn_masks

    def load_data(self, file_path):
        def _create_graph_prompt(graph_data: pd.DataFrame, triplets: pd.DataFrame, grounded_entities: List[str], E=10, N=25):
            result = []
            # Filter the DataFrame 'graph_data' to include only rows where 'code' is in list 'grounded_entities'
            filtered_graph_data = graph_data[graph_data['code'].isin(grounded_entities)]
            if len(filtered_graph_data) > E:
                filtered_graph_data = filtered_graph_data.sample(n=E)

            for row in filtered_graph_data.itertuples(index=False):
                neighbors = row.neighbors
                if len(neighbors) > N:
                    neighbors = random.sample(neighbors, N)
                selected_rows = triplets.loc[(triplets['a'] == row.code) & (triplets['b_text'].isin(neighbors)), ["a_text", "rel", "b_text"]]
                if len(selected_rows) > 0:
                    row_triplets_text = ", ".join(f"{{{a_text}, {rel}, {b_text}}}" for a_text, rel, b_text in selected_rows.itertuples(index=False))
                    result.append(row_triplets_text)
            result = "[" + ", ".join(result) + "]"
            return result

        def calc_stuff(messages, replace_kg_embeddings):
            # Loop for batch tokenization
            question_ids = {}
            input_ids = {}
            attn_masks = {}
            labels = []
            label_ids = []
            user_prompts = []
            if self.graph_in_prompt is not None:
                print(f"Loading {self.graph_in_prompt}")
                graphdf = pd.read_json(self.graph_in_prompt, lines=True)
                if self.is_inference:
                    print(f"Loading {os.path.dirname(self.graph_in_prompt)}/test_triplets.jsonl")
                    testtripletsdf = pd.read_json(f"{os.path.dirname(self.graph_in_prompt)}/test_triplets.jsonl", lines=True)
                else:
                    print(f"Loading {os.path.dirname(self.graph_in_prompt)}/triplets.jsonl")
                    tripletsdf=pd.read_json(f"{os.path.dirname(self.graph_in_prompt)}/triplets.jsonl", lines=True)

            i=0
            for id_, m in tqdm(messages.items(), leave=False, total=len(messages), desc="Building dataset"):
                prompt = m[0]["content"]
                if self.graph_in_prompt is not None:
                    if self.is_inference:
                        # shortcut -- but need to control for number of entities and number of neighbours 
                        triplets_text = testtripletsdf.loc[testtripletsdf['question_id'] == id_, 'kg_content'].values[0]
                    else:
                        grounded_entities = grounded[id_]["qc"]
                        triplets_text = _create_graph_prompt(graphdf, tripletsdf, grounded_entities, N=2)
                    
                    prompt = prompt.replace("given concepts", "given triplets {object, predicate, subject}")
                    if replace_kg_embeddings:
                        prompt = prompt.replace("{{kg_embedding}}", triplets_text)
                    else:
                        prompt = prompt.replace("{kg_embedding}", "{kg_embedding} Triplets: " + triplets_text)
                    m[0]["content"] = prompt
                    sentence = prompt.lower().split("input: ")[1]
                else:
                    sentence = prompt.lower().split("input: ")[1]

                question_ids["".join(sentence.split())] = id_
                _user_prompt, _input_ids, _attn_masks = self.tokenize_and_apply_template(m)
                user_prompts.append(_user_prompt)
                input_ids[i] = _input_ids
                attn_masks[i] = _attn_masks

                text_label = m[-1]["content"]
                label_ids.append(torch.tensor(self.label2id[text_label[:2]]))
                label_input_ids = self.tokenizer.encode(
                    text_label, return_tensors="pt", padding="max_length"
                )
                labels.append(label_input_ids.squeeze(0))
                i += 1
            return user_prompts, question_ids, input_ids, attn_masks, labels, label_ids

        messages = load_jsonl_file(file_path, "dict", only_letter=False)
        # messages = dict(list(messages.items())[:10])
        grounded = load_jsonl_file(f"{os.path.dirname(file_path)}/{self.split}_grounding.jsonl", "dict")
        user_prompts, question_ids, input_ids, attn_masks, labels, label_ids = calc_stuff(messages, self.embeddings_dir == None)

        return user_prompts, question_ids, input_ids, attn_masks, labels, label_ids, grounded

    def load_embeddings(self, dtype, embeddings_path, max_num_embeds=20):
        if embeddings_path.endswith(".pkl"):
            data_dict = pickle.load(open(embeddings_path, "rb"))
        else:
            assert embeddings_path.endswith(".h5")
            data_dict = load_h5_file(embeddings_path, unsqueeze=False)
        embed_dim=list(data_dict.values())[0].shape[0]
        print(f"Embeddings vocab. size: {len(data_dict)}")
    
        all_embeddings = {}
        print("Building prefix for each input prompt to accelerate training")
        counter = 0
        if self.umls_parents is not None:
            max_num_embeds = 3 * max_num_embeds
            
        for i, this_id in enumerate(self.question_ids.values()):
            # Retrieve embeddings
            concept_ids = self.grounded[this_id]["qc"]
            if len(concept_ids) > max_num_embeds:
                random.shuffle(concept_ids)
                concept_ids = concept_ids[:max_num_embeds]

            ## Add parents (2 per entity)
            if self.umls_parents is not None:
                original_concept_ids = concept_ids
                concept_ids = []
                for cui in original_concept_ids:
                    parents = self.umls_parents[cui]
                    if parents:
                        random.shuffle(parents)
                        j=0
                        n_parents = 0
                        while j < len(parents):
                            if parents[j] in data_dict:
                                concept_ids.append( parents[j] )
                                n_parents += 1
                                if n_parents == 2:
                                    break
                            j += 1
                        concept_ids.append(cui)

            embeds = [data_dict[cid] for cid in concept_ids if cid in data_dict]

            if len(embeds) == 0:
                all_embeddings[i] = torch.zeros((max_num_embeds, embed_dim), dtype=dtype)
                counter += 1 
            elif len(embeds) > max_num_embeds:
                all_embeddings[i] = torch.stack(embeds[len(embeds)-max_num_embeds:])
            elif len(embeds) < max_num_embeds:
                padding = torch.zeros((max_num_embeds - len(embeds), embed_dim), dtype=dtype)
                all_embeddings[i] = torch.cat((padding, torch.stack(embeds)), dim=0)
            else:
                all_embeddings[i] = torch.stack(embeds)

            assert len(all_embeddings[i].shape) == 2

        print(f"{counter} examples with no embeddings")
        return all_embeddings
        

class PubMedQADataset(Dataset):
    def __init__(self, args, dtype, tokenizer, normalize_prefix, prefix_length, random_prefix, is_training, is_inference=False):
        self.modelname = args.modelname
        self.normalize_prefix = normalize_prefix
        self.prefix_length = prefix_length
        self.random_prefix = random_prefix
        self.is_training = is_training
        self.is_inference = is_inference
        self.graph_in_prompt = args.graph_in_prompt
        self.embeddings_dir = args.embeddings_dir
        self.tokenizer = tokenizer
        self.max_num_embeds = 20

        # FOR MISTRAL-INSTRUCT MODELS
        self.instruction_template = {
            "meg-mistral-instruct-v0.1": [733, 16289, 28793],
            "meg-mistral-instruct-v0.3": [3],
            "mistral-instruct-v0.1": [733, 16289, 28793],
            "mistral-instruct-v0.3": [3]
        }
    
        self.label2id = {"A)": 0, "B)": 1, "C)": 2}
        self.id2label = {v: k for k, v in self.label2id.items()}

        if is_training:
            self.split = "train"
            data_path = args.data_path
        else:
            if is_inference: 
                self.split = "test"
                data_path = args.test_data_path
            else: 
                self.split = "dev"
                data_path = args.eval_data_path

        (
            self.user_prompts,
            self.question_ids,
            self.input_ids,
            self.attn_masks,
            self.labels,
            self.label_ids,
            self.grounded 
        ) = self.load_data(data_path)

        self.prefixes = None
        if self.embeddings_dir is not None:
            self.prefixes = self.load_embeddings(dtype, args.embeddings_dir, max_num_embeds=self.max_num_embeds)

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, ...]:
        prefix = []
        if self.prefixes is not None:
            if self.random_prefix:
                rnd_id = random.choice(list(self.input_ids.keys()))
                prefix = self.prefixes[rnd_id]  # (prefix_dim)
            else:
                prefix = self.prefixes[ i ]  # (prefix_dim)
            if self.normalize_prefix:
                prefix = prefix.float()
                prefix = prefix / prefix.norm(2, -1)
            # prefix = prefix.unsqueeze(0)

        output = dict(
            qid=i,
            input_ids=self.input_ids[i],
            attention_mask=self.attn_masks[i],
            labels=self.labels[i],
            label_ids=self.label_ids[i],
            prefix=prefix
        )  
        return output

    def tokenize_and_apply_template(self, conv):
        """
        conv=[{'role': 'user',
               'content': "Please, address ..."},
              {'role': 'assistant',
               'content': "A) Negative for the patient's health"}]
        """
    
        if not self.is_inference:
            _user_prompt = conv[0]["content"]
            tokenizer_out = self.tokenizer.apply_chat_template(
                conv, return_tensors="pt", padding="max_length", return_dict=True
            )
        else:
            conv_formatted = self.tokenizer.apply_chat_template(conv, tokenize=False)
            # Remove the initial <s> ([3:]) as it will be added by the tokenizer again later
            _user_prompt = conv_formatted.split("[/INST]")[0][4:] + "[/INST]"
            tokenizer_out = self.tokenizer(
                _user_prompt, return_tensors="pt", padding="max_length", return_attention_mask=True
            )
        _input_ids = tokenizer_out.input_ids.squeeze(0)
        _attn_masks = tokenizer_out.attention_mask.squeeze(0)
        # Truncate if necessary
        bs = _input_ids.shape[0]
        if _input_ids.shape[-1] > self.tokenizer.model_max_length:
            if self.tokenizer.truncation_side == "right":
                # 1. truncate
                _input_ids = _input_ids[: self.tokenizer.model_max_length]
                _attn_masks = _attn_masks[: self.tokenizer.model_max_length]
            else:
                # 1. truncate
                _input_ids = _input_ids[
                    -(
                        self.tokenizer.model_max_length
                        - len(self.instruction_template[self.modelname])
                    ) :
                ]
                _attn_masks = _attn_masks[
                    -(
                        self.tokenizer.model_max_length
                        - len(self.instruction_template[self.modelname])
                    ) :
                ]
                # 2. add instruction
                _input_ids = torch.cat(
                    (torch.tensor(self.instruction_template[self.modelname]), _input_ids)
                )
                _attn_masks = torch.cat(
                    (
                        torch.ones(len(self.instruction_template[self.modelname]), dtype=torch.int),
                        _attn_masks,
                    )
                )

        return _user_prompt, _input_ids, _attn_masks

    def load_data(self, file_path):
        def calc_stuff(messages):
            # Loop for batch tokenization
            question_ids = {}
            input_ids = {}
            attn_masks = {}
            labels = {}
            label_ids = {}
            user_prompts = {}

            i=0
            for id_, m in tqdm(messages.items(), leave=False, total=len(messages), desc="Building dataset"):
                sentence = m[0]["content"].lower().split("context: ")[1]

                key_sentence = "".join(sentence.split())
                if key_sentence not in question_ids:
                    question_ids[key_sentence] = id_
                    _user_prompt, _input_ids, _attn_masks = self.tokenize_and_apply_template(m)
                    user_prompts[i] = _user_prompt
                    input_ids[i] = _input_ids
                    attn_masks[i] = _attn_masks

                    text_label = m[-1]["content"]
                    label_ids[i] = torch.tensor(self.label2id[text_label[:2]])
                    label_input_ids = self.tokenizer.encode(
                        text_label, return_tensors="pt", padding="max_length"
                    )
                    labels[i] = label_input_ids.squeeze(0)
                    i+=1
            return user_prompts, question_ids, input_ids, attn_masks, labels, label_ids

        messages = load_jsonl_file(file_path, "dict", only_letter=False)
        file_dir = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)
        grounded = load_jsonl_file(f"{file_dir}/{file_name.replace('with_graph_embeds_no_marker_end', 'grounding')}", "dict")
        user_prompts, question_ids, input_ids, attn_masks, labels, label_ids = calc_stuff(messages)

        return user_prompts, question_ids, input_ids, attn_masks, labels, label_ids, grounded

    def load_embeddings(self, dtype, embeddings_path, max_num_embeds=20):
        if embeddings_path.endswith(".pkl"):
            data_dict = pickle.load(open(embeddings_path, "rb"))
        else:
            assert embeddings_path.endswith(".h5")
            with h5py.File(embeddings_path, "r") as f:
                data_dict = {}
                for key in f.keys():
                    data_dict[key.replace("___", "/").replace("__", " ")] = torch.tensor(f[key][:])
            if "p_75" in data_dict:
                data_dict['p75'] = data_dict.pop('p_75')
        data_dict = {key: value.to(dtype) for key, value in data_dict.items()}
        embed_dim=list(data_dict.values())[0].shape[0]
        print(f"Embeddings vocab. size: {len(data_dict)}")
    
        all_embeddings = {}
        print("Building prefix for each input prompt to accelerate training")
        counter = 0
        for i, this_id in enumerate(self.question_ids.values()):
            # Retrieve embeddings
            concept_ids = self.grounded[this_id]["qc"]
            embeds = [data_dict[cid] for cid in concept_ids if cid in data_dict]

            if len(embeds) == 0:
                all_embeddings[i] = torch.zeros((max_num_embeds, embed_dim), dtype=dtype)
                counter += 1 
            elif len(embeds) > max_num_embeds:
                random.shuffle(embeds)
                all_embeddings[i] = torch.stack(embeds[:max_num_embeds])
            elif len(embeds) < max_num_embeds:
                padding = torch.zeros((max_num_embeds - len(embeds), embed_dim), dtype=dtype)
                all_embeddings[i] = torch.cat((padding, torch.stack(embeds)), dim=0)
            else:
                all_embeddings[i] = torch.stack(embeds)

        print(f"{counter} examples with no embeddings")
        return all_embeddings


class MedMCQADataset(Dataset):
    def __init__(self, args, dtype, tokenizer, normalize_prefix, prefix_length, random_prefix, is_training, is_inference=False):
        self.modelname = args.modelname
        self.normalize_prefix = normalize_prefix
        self.prefix_length = prefix_length
        self.random_prefix = random_prefix
        self.is_training = is_training
        self.is_inference = is_inference
        self.graph_in_prompt = args.graph_in_prompt
        self.embeddings_dir = args.embeddings_dir
        self.umls_parents =  load_jsonl_file(args.umls_parents, "dict")
        self.tokenizer = tokenizer
        self.max_num_embeds = args.max_num_embeds
        
        # FOR MISTRAL-INSTRUCT MODELS
        self.instruction_template = {
            "meg-mistral-instruct-v0.1": [733, 16289, 28793],
            "meg-mistral-instruct-v0.3": [3],
            "mistral-instruct-v0.1": [733, 16289, 28793],
            "mistral-instruct-v0.3": [3]
        }
    
        self.label2id = {"A)": 0, "B)": 1, "C)": 2,  "D)": 3}
        self.id2label = {v: k for k, v in self.label2id.items()}

        if is_training:
            self.split = "train"
            data_path = args.data_path
        else:
            if is_inference: 
                self.split = "test"
                data_path = args.test_data_path
            else: 
                self.split = "dev"
                data_path = args.eval_data_path

        (
            self.user_prompts,
            self.question_ids,
            self.input_ids,
            self.attn_masks,
            self.labels,
            self.label_ids,
            self.grounded 
        ) = self.load_data(data_path)

        self.prefixes = None
        
        if self.embeddings_dir is not None:
            self.prefixes = self.load_embeddings(dtype, args.embeddings_dir, max_num_embeds=self.max_num_embeds)

        print("input_ids", len(self.input_ids))
        print("question_ids", len(self.question_ids))
        print("prefixes", len(self.prefixes))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, ...]:
        prefix = []
        if self.prefixes is not None:
            if self.random_prefix:
                prefix = torch.rand(self.prefixes[i].shape, dtype=torch.float16)
            else:
                prefix = self.prefixes[ i ]  # (prefix_dim)

            if self.normalize_prefix:
                # hinsage seems to give normalized node embeddings 
                prefix = prefix.float()
                prefix = prefix / prefix.norm(2, -1)

        output = dict(
            qid=i,
            input_ids=self.input_ids[i],
            attention_mask=self.attn_masks[i],
            labels=self.labels[i],
            label_ids=self.label_ids[i],
            prefix=prefix
        )  
        return output

    def tokenize_and_apply_template(self, conv):
        """
        conv=[{'role': 'user',
               'content': "Please, address ..."},
              {'role': 'assistant',
               'content': "A) Negative for the patient's health"}]
        """
    
        if not self.is_inference:
            _user_prompt = conv[0]["content"]
            tokenizer_out = self.tokenizer.apply_chat_template(
                conv, return_tensors="pt", padding="max_length", return_dict=True
            )
        else:
            conv_formatted = self.tokenizer.apply_chat_template(conv, tokenize=False)
            # Remove the initial <s> ([3:]) as it will be added by the tokenizer again later
            _user_prompt = conv_formatted.split("[/INST]")[0][4:] + "[/INST]"
            tokenizer_out = self.tokenizer(
                _user_prompt, return_tensors="pt", padding="max_length", return_attention_mask=True
            )
        _input_ids = tokenizer_out.input_ids.squeeze(0)
        _attn_masks = tokenizer_out.attention_mask.squeeze(0)
        # Truncate if necessary
        bs = _input_ids.shape[0]
        if _input_ids.shape[-1] > self.tokenizer.model_max_length:
            if self.tokenizer.truncation_side == "right":
                # 1. truncate
                _input_ids = _input_ids[: self.tokenizer.model_max_length]
                _attn_masks = _attn_masks[: self.tokenizer.model_max_length]
            else:
                # 1. truncate
                _input_ids = _input_ids[
                    -(
                        self.tokenizer.model_max_length
                        - len(self.instruction_template[self.modelname])
                    ) :
                ]
                _attn_masks = _attn_masks[
                    -(
                        self.tokenizer.model_max_length
                        - len(self.instruction_template[self.modelname])
                    ) :
                ]
                # 2. add instruction
                _input_ids = torch.cat(
                    (torch.tensor(self.instruction_template[self.modelname]), _input_ids)
                )
                _attn_masks = torch.cat(
                    (
                        torch.ones(len(self.instruction_template[self.modelname]), dtype=torch.int),
                        _attn_masks,
                    )
                )
        
        return _user_prompt, _input_ids, _attn_masks

    def load_data(self, file_path):
        def _create_graph_prompt(graph_data: pd.DataFrame, triplets: pd.DataFrame, grounded_entities: List[str], E=10, N=25):
            result = []
            # Filter the DataFrame 'graph_data' to include only rows where 'code' is in list 'grounded_entities'
            filtered_graph_data = graph_data[graph_data['code'].isin(grounded_entities)]
            if len(filtered_graph_data) > E:
                filtered_graph_data = filtered_graph_data.sample(n=E)

            for row in filtered_graph_data.itertuples(index=False):
                neighbors = row.neighbors
                if len(neighbors) > N:
                    neighbors = random.sample(neighbors, N)
                selected_rows = triplets.loc[(triplets['a'] == row.code) & (triplets['b_text'].isin(neighbors)), ["a_text", "rel", "b_text"]]
                if len(selected_rows) > 0:
                    row_triplets_text = ", ".join(f"{{{a_text}, {rel}, {b_text}}}" for a_text, rel, b_text in selected_rows.itertuples(index=False))
                    result.append(row_triplets_text)
            result = "[" + ", ".join(result) + "]"
            return result

        def calc_stuff(messages, replace_kg_embeddings):
            # Loop for batch tokenization
            question_ids = {}
            input_ids = {}
            attn_masks = {}
            label_ids = {}
            labels = {}
            user_prompts = []
            if self.graph_in_prompt is not None:
                print(f"Loading {self.graph_in_prompt}")
                graphdf = pd.read_json(self.graph_in_prompt, lines=True)
                if self.is_inference:
                    print(f"Loading {os.path.dirname(self.graph_in_prompt)}/test_triplets.jsonl")
                    testtripletsdf = pd.read_json(f"{os.path.dirname(self.graph_in_prompt)}/test_triplets.jsonl", lines=True)
                else:
                    print(f"Loading {os.path.dirname(self.graph_in_prompt)}/triplets.jsonl")
                    tripletsdf=pd.read_json(f"{os.path.dirname(self.graph_in_prompt)}/triplets.jsonl", lines=True)

            i=0
            duplicates = []
            for id_, m in tqdm(messages.items(), leave=False, total=len(messages), desc="Building dataset"):
                prompt = m[0]["content"]
                if self.graph_in_prompt is not None:
                    if self.is_inference:
                        # shortcut -- but need to control for number of entities and number of neighbours 
                        triplets_text = testtripletsdf.loc[testtripletsdf['question_id'] == id_, 'kg_content'].values[0]
                    else:
                        grounded_entities = grounded[id_]["qc"]
                        triplets_text = _create_graph_prompt(graphdf, tripletsdf, grounded_entities, N=2)
                    
                    prompt = prompt.replace("given concepts", "given triplets {object, predicate, subject}")
                    if replace_kg_embeddings:
                        prompt = prompt.replace("{{kg_embedding}}", triplets_text)
                    else:
                        prompt = prompt.replace("{kg_embedding}", "{kg_embedding} Triplets: " + triplets_text)
                    m[0]["content"] = prompt
                    sentence = prompt.lower().split("input: ")[1]
                else:
                    sentence = prompt.lower().split("question: ")[1]

                key_sentence = "".join(sentence.split())
                if key_sentence in question_ids:
                    duplicates.append( (question_ids[key_sentence], id_) )
                else:
                    question_ids[key_sentence] = id_
                    _user_prompt, _input_ids, _attn_masks = self.tokenize_and_apply_template(m)
                    user_prompts.append(_user_prompt)
                    input_ids[i] = _input_ids
                    attn_masks[i] = _attn_masks

                    text_label = m[-1]["content"]
                    label_ids[i] = torch.tensor(self.label2id[text_label[:2]])
                    label_input_ids = self.tokenizer.encode(
                        text_label, return_tensors="pt", padding="max_length"
                    )
                    labels[i] = label_input_ids.squeeze(0)
                    i += 1
            print(f"Detected {len(duplicates)} duplicates")
            return user_prompts, question_ids, input_ids, attn_masks, labels, label_ids

        messages = load_jsonl_file(file_path, "dict", only_letter=False)
        # messages = dict(list(messages.items())[:10])
        file_dir = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)
        grounded = load_jsonl_file(f"{file_dir}/{file_name.replace('with_graph_embeds_no_marker_end', 'grounding')}", "dict")
        user_prompts, question_ids, input_ids, attn_masks, labels, label_ids = calc_stuff(messages, self.embeddings_dir == None)
        # assert len(question_ids) == len(input_ids) == len(labels) == len(label_ids)
        return user_prompts, question_ids, input_ids, attn_masks, labels, label_ids, grounded

    def load_embeddings(self, dtype, embeddings_path, max_num_embeds=20):
        if embeddings_path.endswith(".pkl"):
            data_dict = pickle.load(open(embeddings_path, "rb"))
        else:
            assert embeddings_path.endswith(".h5")
            data_dict = load_h5_file(embeddings_path, unsqueeze=False)
        embed_dim=list(data_dict.values())[0].shape[0]
        print(f"Embeddings vocab. size: {len(data_dict)}")
    
        all_embeddings = {}
        print("Building prefix for each input prompt to accelerate training")
        counter = 0
        if self.umls_parents is not None:
            max_num_embeds = 3 * max_num_embeds
            
        for i, this_id in enumerate(self.question_ids.values()):
            # Retrieve embeddings
            concept_ids = self.grounded[this_id]["qc"]
            if len(concept_ids) > max_num_embeds:
                random.shuffle(concept_ids)
                concept_ids = concept_ids[:max_num_embeds]

            ## Add parents (2 per entity)
            if self.umls_parents is not None:
                original_concept_ids = concept_ids
                concept_ids = []
                for cui in original_concept_ids:
                    parents = self.umls_parents[cui]
                    if parents:
                        random.shuffle(parents)
                        j=0
                        n_parents = 0
                        while j < len(parents):
                            if parents[j] in data_dict:
                                concept_ids.append( parents[j] )
                                n_parents += 1
                                if n_parents == 2:
                                    break
                            j += 1
                        concept_ids.append(cui)

            embeds = [data_dict[cid] for cid in concept_ids if cid in data_dict]

            if len(embeds) == 0:
                all_embeddings[i] = torch.zeros((max_num_embeds, embed_dim), dtype=dtype)
                counter += 1 
            elif len(embeds) > max_num_embeds:
                all_embeddings[i] = torch.stack(embeds[len(embeds)-max_num_embeds:])
            elif len(embeds) < max_num_embeds:
                padding = torch.zeros((max_num_embeds - len(embeds), embed_dim), dtype=dtype)
                all_embeddings[i] = torch.cat((padding, torch.stack(embeds)), dim=0)
            else:
                all_embeddings[i] = torch.stack(embeds)

            assert len(all_embeddings[i].shape) == 2

        print(f"{counter} examples with no embeddings")
        return all_embeddings


class UMLSMappingDataset(Dataset):
    """
    Dataset loading content from args.data_path=vocab_training_meg.jsonl if the form of
    [INST] Explain to me this medical concept {kg_embedding} [/INST] 2,4-dichlorophenoxyacetic acid
    """
    def __init__(self, args, dtype, tokenizer, normalize_prefix, prefix_length, random_prefix, is_training, is_inference=False):        
        self.modelname = args.modelname
        self.normalize_prefix = normalize_prefix
        self.prefix_length = prefix_length
        self.random_prefix = random_prefix
        self.max_seq_len = args.model_max_length
        self.is_training = is_training
        self.is_inference = is_inference
        self.tokenizer = tokenizer

        # FOR MISTRAL-INSTRUCT MODELS
        self.instruction_template = {
            "meg-mistral-instruct-v0.1": [733, 16289, 28793],
            "meg-mistral-instruct-v0.3": [3]
        }

        if is_training:
            self.split = "train"
            data_path = args.data_path
        else:
            if is_inference: 
                self.split = "test"
                data_path = args.test_data_path
            else: 
                self.split = "dev"
                data_path = args.eval_data_path

        self.kg_embeddings = self.load_embeddings(dtype, args.embeddings_dir)

        (
            self.input_ids,
            self.attn_masks,
            self.labels,
            self.labels_text,
            self.code2name_df,
            self.id2code
        ) = self.load_data(data_path)

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, ...]:
        code = self.code2name_df.loc[self.code2name_df["name"] == self.labels_text[i], "code"].values[0] 
        return dict(
            qid=i,
            input_ids=self.input_ids[i],
            attention_mask=self.attn_masks[i],
            labels=self.labels[i],
            label_ids=self.labels[i],
            prefix=self.kg_embeddings[code]
        )


    def tokenize_and_apply_template(self, conv):
        """
        conv=[{'role': 'user',
               'content': "Please, address ..."},
              {'role': 'assistant',
               'content': "A) Negative for the patient's health"}]
        """
    
        if not self.is_inference:
            _user_prompt = conv[0]["content"]
            tokenizer_out = self.tokenizer.apply_chat_template(
                conv, return_tensors="pt", padding="max_length", return_dict=True
            )
        else:
            conv_formatted = self.tokenizer.apply_chat_template(conv, tokenize=False)
            # Remove the initial <s> ([3:]) as it will be added by the tokenizer again later
            _user_prompt = conv_formatted.split("[/INST]")[0][4:] + "[/INST]"
            tokenizer_out = self.tokenizer(
                _user_prompt, return_tensors="pt", padding="max_length", return_attention_mask=True
            )
        _input_ids = tokenizer_out.input_ids.squeeze(0)
        _attn_masks = tokenizer_out.attention_mask.squeeze(0)
        # Truncate if necessary
        if len(_input_ids) > self.tokenizer.model_max_length:
            # 1. truncate from right side as the [INST] blahblah [/INST] prompt is fixed 
            _input_ids = _input_ids[: self.tokenizer.model_max_length]
            _attn_masks = _attn_masks[: self.tokenizer.model_max_length]

        return _user_prompt, _input_ids, _attn_masks

    def load_data(self, file_path):
        messages = load_jsonl_file(file_path, "dict")
        code2name = pd.read_csv(f"{os.path.dirname(file_path)}/vocab.csv", sep="\t", names=["code", "name"])
        code2name = self.adjust_vocab_to_embeddings(code2name)
        entity_names = set(code2name.name.values)

        str_pattern_length = len("{kg_embedding}")
        input_ids={}
        attn_masks={}
        labels={}
        labels_text={}
        id2code={}
        c=0
        id_ = 0
        for m in tqdm(messages.values(), leave=False, total=len(messages), desc="Processing messages"):
            target_text = m[-1]["content"]
            if target_text not in entity_names:
                continue
            m[0]["content"] = m[0]["content"][:-str_pattern_length]
            _user_prompt, _input_ids, _attn_masks = self.tokenize_and_apply_template(m)

            # Assign values
            id2code[id_] = code2name.loc[code2name["name"] == target_text, "code"].values
            input_ids[id_] = _input_ids
            attn_masks[id_] = _attn_masks
            labels_text[id_] = target_text
            labels[id_] = _input_ids.clone()
            # Increment index
            id_ += 1
            # c+=1
            # if c==100:
            #     break
        # print(f"{discard} samples discarded")
        print("User:", _user_prompt)
        print("Assistant:", target_text)
        print(f"{len(input_ids)}/{len(messages)} mapping loaded")
        sys.stdout.flush()
        assert len(input_ids) == len(attn_masks) == len(labels_text)
        return input_ids, attn_masks, labels, labels_text, code2name, id2code

    def load_embeddings(self, dtype, embeddings_path):
        if embeddings_path.endswith(".pkl"):
            data_dict = pickle.load(open(embeddings_path, "rb"))
        else:
            assert embeddings_path.endswith(".h5")
            data_dict = load_h5_file(embeddings_path)
        data_dict = {key: value.to(dtype) for key, value in data_dict.items()}
        print(f"Embeddings vocab. size: {len(data_dict)}")
        return data_dict
    
    def adjust_vocab_to_embeddings(self, code2name_df):
        if len(self.kg_embeddings) < len(code2name_df):
            code2name_df = code2name_df[
                code2name_df['code'].isin(self.kg_embeddings.keys())
                ]
            print(f"Vocab size reduced to {len(code2name_df)}")
        return code2name_df


class UMLSMappingDatasetExtended(Dataset):
    """
    Dataset loading content from args.data_path=vocab_training_meg.jsonl if the form of
    [INST] Explain to me these medical concepts {kg_embedding} [/INST] 2,4-dichlorophenoxyacetic acid, blah, blah
    """
    def __init__(self, args, dtype, tokenizer, normalize_prefix, prefix_length, random_prefix, is_training, is_inference=False):        
        self.modelname = args.modelname
        self.normalize_prefix = normalize_prefix
        self.prefix_length = prefix_length
        self.random_prefix = random_prefix
        self.max_seq_len = args.model_max_length
        self.is_training = is_training
        self.is_inference = is_inference
        self.tokenizer = tokenizer

        # FOR MISTRAL-INSTRUCT MODELS
        self.instruction_template = {
            "meg-mistral-instruct-v0.1": [733, 16289, 28793],
            "meg-mistral-instruct-v0.3": [3]
        }

        if is_training:
            self.split = "train"
            data_path = args.data_path
        else:
            if is_inference: 
                self.split = "test"
                data_path = args.test_data_path
            else: 
                self.split = "dev"
                data_path = args.eval_data_path

        (
            self.input_ids,
            self.attn_masks,
            self.labels,
            self.labels_text,
            self.code2name_df,
            self.input_codes
        ) = self.load_data(data_path)

        self.kg_embeddings = self.load_embeddings(dtype, args.embeddings_dir)

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, ...]:
        # code = self.code2name_df.loc[self.code2name_df["name"] == self.labels_text[i], "code"].values[0]
        # embeddings = [ self.kg_embeddings[code] for code in self.input_codes[i] ]
        return dict(
            qid=i,
            input_ids=self.input_ids[i],
            attention_mask=self.attn_masks[i],
            labels=self.labels[i],
            label_ids=self.labels[i],
            prefix=self.kg_embeddings[i]
        )

    def tokenize_and_apply_template(self, conv):
        """
        conv=[{'role': 'user',
               'content': "Please, address ..."},
              {'role': 'assistant',
               'content': "A) Negative for the patient's health"}]
        """
        if "mistral-instruct-v0.1" in self.modelname:
            if not self.is_inference:
                _user_prompt = conv[0]["content"]
                tokenizer_out = self.tokenizer.apply_chat_template(
                    conv, return_tensors="pt", padding="max_length", return_dict=True
                )
            else:
                conv_formatted = self.tokenizer.apply_chat_template(conv, tokenize=False)
                # Remove the initial <s> ([3:]) as it will be added by the tokenizer again later
                _user_prompt = conv_formatted.split("[/INST]")[0][4:] + "[/INST]"
                tokenizer_out = self.tokenizer(
                    _user_prompt, return_tensors="pt", padding="max_length", return_attention_mask=True
                )
            _input_ids = tokenizer_out.input_ids.squeeze(0)
            _attn_masks = tokenizer_out.attention_mask.squeeze(0)
            # Truncate if necessary
            if len(_input_ids) > self.tokenizer.model_max_length:
                # 1. truncate from right side as the [INST] blahblah [/INST] prompt is fixed 
                _input_ids = _input_ids[: self.tokenizer.model_max_length]
                _attn_masks = _attn_masks[: self.tokenizer.model_max_length]
        else:
            raise Exception(
                f"Template from model {self.modelname} is not currently supported"
            )
        return _user_prompt, _input_ids, _attn_masks

    def load_data(self, file_path):
        messages = load_jsonl_file(file_path, "dict")
        name2code= {}
        max_input_len = []
        with open(f"{os.path.dirname(file_path)}/vocab.csv", mode='r') as file:
            reader = csv.reader(file, delimiter='\t')
            for row in reader:
                # Assume the file has exactly two columns per row
                key, value = row
                name2code[value] = key

        str_pattern_length = len("{kg_embedding}")
        input_ids={}
        input_codes={}
        attn_masks={}
        labels={}
        labels_text={}
        c=0
        id_ = 0
        for m in tqdm(messages.values(), leave=False, total=len(messages), desc="Processing messages"):
            target_text = m[-1]["content"]
            m[0]["content"] = m[0]["content"][:-str_pattern_length]
            _user_prompt, _input_ids, _attn_masks = self.tokenize_and_apply_template(m)

            # Assign values
            max_input_len.append(len(_input_ids))
            input_ids[id_] = _input_ids
            attn_masks[id_] = _attn_masks
            labels_text[id_] = target_text
            labels[id_] = _input_ids.clone()

            input_codes[id_] = [name2code.get(name.rstrip().lstrip(), -1) for name in target_text.split(";")]
            # Increment index
            id_ += 1
            # c+=1
            # if c==100:
            #     break
        # print(f"{discard} samples discarded")
        print("User:", _user_prompt)
        print("Assistant:", target_text)
        print(f"{len(input_ids)}/{len(messages)} mapping loaded")
        sys.stdout.flush()
        return input_ids, attn_masks, labels, labels_text, name2code, input_codes

    def load_embeddings(self, dtype, embeddings_path):
        if embeddings_path.endswith(".pkl"):
            data_dict = pickle.load(open(embeddings_path, "rb"))
        else:
            assert embeddings_path.endswith(".h5")
            data_dict = load_h5_file(embeddings_path)
        data_dict = {key: value.to(dtype) for key, value in data_dict.items()}
        print(f"Embeddings vocab. size: {len(data_dict)}")

        all_embeddings = {}
        max_num_embeds=10
        embed_dim=list(data_dict.values())[0].shape[1]
        print("Building prefix for each input prompt to accelerate training")
        counter = 0
        for i, concept_ids in enumerate(self.input_codes.values()):
            # Retrieve embeddings
            embeds = [data_dict[cid].squeeze(0) for cid in concept_ids if cid in data_dict]

            if len(embeds) == 0:
                all_embeddings[i] = torch.zeros((max_num_embeds, embed_dim), dtype=dtype)
                counter += 1 
            elif len(embeds) > max_num_embeds:
                random.shuffle(embeds)
                all_embeddings[i] = torch.stack(embeds[:max_num_embeds])
            elif len(embeds) < max_num_embeds:
                padding = torch.zeros((max_num_embeds - len(embeds), embed_dim), dtype=dtype)
                all_embeddings[i] = torch.cat((padding, torch.stack(embeds)), dim=0)
            else:
                all_embeddings[i] = torch.stack(embeds)

        print(f"{counter} examples with no embeddings")
        return all_embeddings
        

def load_custom_dataset(data_args, model_args, tokenizer, dtype, is_train, is_inference=False):
    datasetClass = {"medqa_usmle": MedQADataset,
                    "pubmedqa": PubMedQADataset,
                    "medmcqa": MedMCQADataset,
                    "mmlu_medical": MedMCQADataset,
                    "umls": UMLSMappingDataset,
                    "umls-extended": UMLSMappingDatasetExtended}[data_args.task]
    dataset = datasetClass(
        data_args, 
        dtype, 
        tokenizer, 
        model_args.normalize_prefix, 
        model_args.prefix_length, 
        model_args.random_prefix, 
        is_training=is_train, 
        is_inference=is_inference
    )
    return dataset


if __name__ == "__main__":
    from argparser import DataArguments

    ROOT = "/placeholder/to/your/data/directory"
    dtype=torch.float16
    normalize_prefix=False
    random_prefix=False
    prefix_length=1

    data_args = DataArguments()
    data_args.task = "medqa_usmle"
    data_args.model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.1"
    data_args.modelname = "mistral-instruct-v0.1"
    ## medqa
    data_args.data_path=f"{ROOT}/medqa_usmle/train.jsonl"
    data_args.test_data_path=f"{ROOT}/medqa_usmle/test.jsonl"
    data_args.embeddings_dir=f"{ROOT}/embeddings/umls/word2embed_graphsage_sapbertinit.h5"
    # data_args.graph_in_prompt=f"{ROOT}/embeddings/umls/neighbors_data_n25.jsonl"

    tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.1",
            cache_dir=None,
            padding=True,
            truncation=True,
            padding_side="left",
            truncation_side="left",
            use_fast=False,
        ) 
    tokenizer.model_max_length = 400
    tokenizer.pad_token = tokenizer.unk_token

    dataset = MedQADataset(data_args, dtype, tokenizer, normalize_prefix, prefix_length, random_prefix, is_training=True)

    print(len(dataset))
    print("Done.")