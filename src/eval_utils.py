import os
import json
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Dict
from collections import Counter
from transformers import EvalPrediction
from sklearn.metrics import classification_report
from custom_datasets import load_custom_dataset
from custom_datasets_llama import load_custom_dataset as load_custom_dataset_llama
from fuzzywuzzy import fuzz

def empty_prefix(p):
    try:
        if not p or p is None or 0 or len(p[0]) == 0:
            return True
    except RuntimeError:
        return torch.equal(p.cpu(), torch.tensor([]).unsqueeze(0))
    return False


def parse_generation_output(    
        generated_texts: List[str],
        answer_classes: List[Dict[int,str]],
        default_class: int = 2,
        model_type="mistral",
        verbose: bool = False,
    ) -> List[int]:
    assert len(answer_classes) == len(generated_texts)

    predictions_id = []
    scores = []
    threshold = 80
    for ii, generated_text in enumerate(generated_texts):
        # initialize defaults 
        highest_score = 0
        best_match = default_class  # None
        generated_text = generated_text.strip()
        if model_type == "mistral" and "[/INST]" in generated_text:
            generated_text = generated_text.split("[/INST]")[-1]
        for class_id, class_text in answer_classes[ii].items():
            if "[/INST]" in class_text:
                class_text = class_text.split("[/INST]")[0]
            score = fuzz.partial_ratio(generated_text, class_text)
            if score == 100:
                highest_score = score
                best_match = class_id
                break
            elif score > highest_score and score >= threshold:
                highest_score = score
                best_match = class_id
        
        predictions_id.append(best_match)
        scores.append(highest_score)
        if verbose:
            if best_match == default_class and score < threshold:
                print(f"Unrecognized answer: {generated_text}")
            elif score < 100:
                print(f"Soft matching (score {score}): {generated_text}")
    
    return predictions_id, scores


def generate_and_decode(task, model, tokenizer, prefix, input_ids, max_new_tokens, device, labels=None):
    if not empty_prefix(prefix):
        prefix = prefix.to(device)
    # Outputs
    batch_output_ids = model.generate(
        task=task,
        prefix=prefix, 
        input_ids=input_ids.to(device), 
        max_new_tokens=max_new_tokens,
        output_scores=True, 
        return_dict_in_generate=False,
        debug=False
    )
    outputs = tokenizer.batch_decode(
        batch_output_ids,
        spaces_between_special_tokens=False,
        skip_special_tokens=True
    )
    # Inputs 
    inputs = tokenizer.batch_decode(
        input_ids,
        spaces_between_special_tokens=False,
        skip_special_tokens=True
    )
    # Labels
    if labels is not None:
        labels = tokenizer.batch_decode(
            labels,
            spaces_between_special_tokens=False,
            skip_special_tokens=True
        )
    return inputs, outputs, labels

    
def custom_collate_fn(batch):
    keys = batch[0].keys()
    collated_batch = {}
    for key in keys:
        if isinstance(batch[0][key], torch.Tensor):
            collated_batch[key] = torch.stack([item[key] for item in batch])
        else:
            collated_batch[key] = [item[key] for item in batch]
    return collated_batch


def delete_file_content(file_path):
    """
    Deletes the content of the specified file.
    
    Parameters:
    file_path (str): The path to the file whose content should be deleted.
    """
    try:
        with open(file_path, 'w') as f:
            pass  
    except FileNotFoundError:
        pass


def do_inference(task, model, tokenizer, test_dataloader, data_args, device, output_file, n_max=None):
    delete_file_content(output_file)

    c=0
    if "umls" in task:
        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                prefix, input_ids, labels = batch["prefix"], batch["input_ids"], batch["labels"]

                inputs, outputs, labels_text = generate_and_decode(task, model, tokenizer, prefix, input_ids, data_args.max_new_tokens, device, labels)

                if dist.get_rank() == 0:
                    # Debugging
                    for ii in range(len(outputs)):
                        ans_json = {
                            "input": inputs[ii],
                            "label": labels_text[ii],
                            "output": outputs[ii], 
                        }
                        print(json.dumps(ans_json))
                # c+=1
                # if c==2: break
    else:
        no_choice=0
        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                qid, prefix, input_ids, label_ids = batch["qid"], batch["prefix"], batch["input_ids"], batch["label_ids"]

                inputs, outputs, _ = generate_and_decode(task, model, tokenizer, prefix, input_ids, data_args.max_new_tokens, device)

                outputs_gen = []
                choices = []
                for output, inp in zip(outputs, inputs):
                    if "[/INST]" in output:
                        outp = output.split("[/INST]")[-1]
                    else:
                        outp = output
                    outputs_gen.append(outp)
                    if "llama3" in data_args.modelname:
                        choices_list = [k.rstrip().lstrip() for k in inp.split("Options:")[-1].split("\nAnswer with the best option directly.")[0].split("\n") 
                                        if k.lstrip().startswith("A)") or k.lstrip().startswith("B)") or k.lstrip().startswith("C)") or k.lstrip().startswith("D)")]
                        if len(choices_list) == 0:
                            no_choice+=1
                            if task == "pubmedqa":
                                choices_list = ["A", "B", "C"]
                                preds="A"
                            else:
                                choices_list = ["A", "B", "C", "D"]
                                preds="A"
                    else:
                        choices_list = [k.rstrip().lstrip() for k in inp.split("Options:")[-1].split("\n") 
                                        if k.lstrip().startswith("A)") or k.lstrip().startswith("B)") or k.lstrip().startswith("C)") or k.lstrip().startswith("D)")]
                    choices.append({i:k for i,k in enumerate(choices_list)})
                preds, scores = parse_generation_output(outputs_gen, choices, default_class=-1)

                if dist.get_rank() == 0:
                    for ii in range(len(outputs)):
                        # Dump answers
                        # os.makedirs(os.path.dirname(output_file), exist_ok=True)
                        with open(os.path.expanduser(output_file), "a+") as fout:
                            ans_json = {
                                "question_id": qid[ii],
                                # "answer_id": shortuuid.uuid(),
                                "model_id": data_args.modelname,
                                "pred": preds[ii],
                                "label": label_ids[ii].item(),
                                "input": inputs[ii],
                                "output": outputs[ii], 
                                "score": scores[ii],
                                "tstamp": time.time()
                            }
                            fout.write(json.dumps(ans_json) + "\n")

                c+=1
                if n_max is not None and c == n_max: break

        print("NO CHOICE:",no_choice)


def main_eval(model, dtype, data_args, training_args, model_args, n_max=None):
    output_file = training_args.output_dir

    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    world_size = int(os.getenv('WORLD_SIZE', '1'))

    tokenizer = model.get_tokenizer()
    model.eval()

    # ## Load dataset & dataloader 
    if "mistral" in data_args.modelname:
        test_dataset = load_custom_dataset(data_args, model_args, tokenizer, dtype, False, True)
    else:
        test_dataset = load_custom_dataset_llama(data_args, model_args, tokenizer, dtype, False, True)
    test_dataloader = DataLoader(test_dataset, 
                                batch_size=training_args.per_device_eval_batch_size, 
                                shuffle=False,
                                drop_last=False,
                                collate_fn=custom_collate_fn)
    
    ## Initialize the DeepSpeed-Inference engine
    device=torch.device('cuda')
    if training_args.deepspeed is not None:
        import deepspeed as ds
        ds_engine = ds.init_inference(model,
                                      dtype=dtype,
                                      tensor_parallel={"tp_size": world_size},
                                      replace_with_kernel_inject=True)
        model = ds_engine.module
    else:
        model.to(device)

    ## Do inference
    do_inference(data_args.task, model, tokenizer, test_dataloader, data_args, device, output_file, n_max)


def compute_classification_report(
    p: EvalPrediction,
    target_names=["Negative", "Positive", "Neutral"],
    groups=None,
    verbose=True,
):
    predictions = p.predictions
    labels = p.label_ids
    results = classification_report(
        labels,
        predictions,
        target_names=target_names,
        output_dict=True,
        zero_division=np.nan,
    )
    result_groups = {}
    if groups is not None and verbose:
        result_groups = {
            str(gid): classification_report(
                predictions[groups == gid],
                labels[groups == gid],
                # target_names=target_names,
                output_dict=True,
                zero_division=np.nan,
            )
            for gid in sorted(np.unique(groups))
        }
    results["entity_type"] = result_groups
    return results


def compute_eval_metrics(task, output_file, seed, stdout=True):
    ## Evaluate
    default_class = -1
    no_prediction = 0
    no_choices = 0
    all_preds, all_labels = [], []
    with open(output_file, "r") as f:
        for l in f:
            line = json.loads(l)
            pred = line["pred"]
            if pred != default_class:
                all_preds.append(pred)
                all_labels.append(line["label"])
            else:
                no_prediction += 1

    preds = EvalPrediction(predictions=all_preds, label_ids=all_labels)
    target_names = ["Negative", "Positive", "Neutral"]
    if task == "medqa_usmle" or task == "medmcqa" or task == "mmlu_medical":
        target_names = ["A", "B", "C", "D"]
    elif task == "pubmedqa":
        target_names = ["A", "B", "C"]
    
    results = compute_classification_report(preds, target_names)
    if stdout == True:
        print(f"Evaluation seed={seed}:")
        print(f"No choices: {no_choices}")
        print(f"No prediction: {no_prediction}")
        print(f"Output distribution: {dict(Counter(all_preds))}")
        print(results)
        print()
    return results
