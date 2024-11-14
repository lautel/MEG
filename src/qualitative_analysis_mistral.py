import os
import logging
import torch
import transformers
from transformers import MistralForCausalLM
from train_meg_mistral import MEGMistralModel
from config import parse_meg_config
from trl import SFTConfig
from argparser import DataArguments, ModelArguments, LoraArguments
from utils import set_seed

logger = logging.getLogger(__name__)
logging.getLogger("transformers.generation_utils").disabled = True
logger.setLevel(logging.DEBUG)

os.environ["TOKENIZERS_PARALLELISM"] = "true"


if __name__ == "__main__":
    parser = transformers.HfArgumentParser(
        (DataArguments, SFTConfig, LoraArguments, ModelArguments)
    )
    data_args, training_args, lora_args, model_args = (
        parser.parse_args_into_dataclasses()
    )
    set_seed(training_args.seed)

    ## Config precision
    dtype = torch.float16
    if training_args.bf16:
        dtype = torch.bfloat16
    else:
        if training_args.tf32:
            dtype = torch.float32

    if data_args.temp < 1e-4:
        do_sample = False
    else:
        do_sample = True

    ## Load MEGMistralModel
    config = parse_meg_config(data_args, training_args, lora_args, model_args, dtype)
    if training_args.resume_from_checkpoint:
        model =  MEGMistralModel.from_pretrained(
            checkpoint_path=training_args.resume_from_checkpoint,
            dtype=dtype,
            config=config,
            data_args=data_args,
            model_args=model_args,
        )
    else:
        model =  MEGMistralModel(config, data_args, model_args)

    tokenizer = model.get_tokenizer()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    question = "What are the first-line pharmacologic treatments for hyperlipidemia in patients with type 2 diabetes mellitus?"
    """
    The first-line pharmacologic treatments for hyperlipidemia in patients with type 2 diabetes mellitus are typically statins, which are medications that lower cholesterol levels in the blood. These medications work by inhibiting the enzyme HMG-CoA reductase, which is involved in the production of cholesterol in the liver. Other medications that may be used to treat hyperlipidemia in patients with type 2 diabetes include bile acid sequestrants, which bind to bile acids in the intestines and prevent them from being absorbed into the bloodstream, and fibrates, which increase the breakdown of triglycerides in the liver. It is important to note that the choice of medication and dosage will depend on the specific needs of the patient and the severity of their hyperlipidemia.
    
    UMLS TUNED
    The first-line pharmacologic treatments for hyperlipidemia in patients with type 2 diabetes mellitus include:\n\n1. Statins: These are the most commonly prescribed medications for hyperlipidemia. They work by inhibiting the production of cholesterol in the liver.\n\n2. Ezetimibe: This medication works by blocking the absorption of cholesterol in the intestines.\n\n3. Bile acid sequestrants: These medications work by binding to bile acids in the intestines and preventing their absorption.\n\n4. Fibrates: These medications work by stimulating the liver to produce more bile acids, which can help lower triglyceride levels.\n\n5. Nicotinic acid: This medication works by reducing the production of very-low-density lipoprotein (VLDL) cholesterol and triglycerides.\n\nIt is important to note that the choice of medication and dosage will depend on the specific type and severity of hyperlipidemia, as well as other factors such as age, medical history, and lifestyle.'
    """

    # question = "Explain to me what you know about Clopidogrel"
    """
    Clopidogrel is a medication used to prevent blood clots. It works by inhibiting the activity of platelets, which are small blood cells that play a key role in blood clotting. Clopidogrel is commonly used to prevent heart attacks and strokes in people who have had these events in the past or who are at high risk for them. It is also used to prevent blood clots in people who have undergone certain types of surgery or who have a condition called atrial fibrillation. Clopidogrel is typically taken as a pill and is usually taken for several weeks or months to prevent blood clots. It is important to follow the dosage instructions provided by your doctor or pharmacist and to inform them of any other medications you are taking, as clopidogrel can interact with some drugs.

    UMLS TUNED
    Clopidogrel is a medication used to prevent blood clots. It works by inhibiting the formation of blood clots by blocking the action of a protein called platelet aggregation factor. Clopidogrel is commonly used in patients with coronary artery disease, stroke, and peripheral artery disease. It is typically taken orally as a tablet or capsule. Common side effects of clopidogrel include bleeding, nausea, vomiting, and diarrhea. It is important to follow the dosage instructions provided by your healthcare provider and to report any side effects to them.
    """

    # question = "What are the active and inactive compounds present in Atorvastatin?"
    """
    Atorvastatin is a medication used to lower cholesterol levels in the blood. It works by inhibiting an enzyme called HMG-CoA reductase, which is involved in the production of cholesterol in the liver. The active compound in Atorvastatin is Atorvastatin itself, while the inactive compounds are Atorvastatin calcium salt and Atorvastatin magnesium salt. These inactive compounds are added to the medication to improve its solubility and absorption in the body.

    UMLS TUNED
    Atorvastatin is a medication used to lower cholesterol levels in the blood. The active compound present in Atorvastatin is Atorvastatin calcium. Atorvastatin calcium is the only active ingredient in Atorvastatin and is responsible for the drug's therapeutic effect. There are no inactive compounds present in Atorvastatin.

    UMLS TUNE + EMBEDDING C0286650
    Atorvastatin is a drug that contains the active compound atorvastatin calcium. It is a cholesterol-lowering medication that works by inhibiting an enzyme called HMG-CoA reductase, which is involved in the production of cholesterol in the liver. The inactive compounds present in Atorvastatin include excipients such as microcrystalline cellulose, croscarmellose sodium, magnesium stearate, and sodium lauryl sulfate. These excipients are added to the medication to help it dissolve, disperse, and stabilize.

    ChatGPT-4o
    Atorvastatin is a medication used primarily for lowering cholesterol and preventing cardiovascular disease. Here's a breakdown of its active and inactive compounds:
    [...] **Summary**
    Active Compound: Atorvastatin Calcium
    Inactive Compounds: Microcrystalline Cellulose, Calcium Phosphate, Croscarmellose Sodium, Magnesium Stearate, Hydroxypropyl Cellulose, Titanium Dioxide, Iron Oxide, Polyethylene Glycol (specific inactive ingredients may vary by formulation).
    """
    tokenizer_out = tokenizer.encode(
        "[INST] " + question + " [/INST]", return_tensors="pt"
    )
    attention_mask = tokenizer_out.ne(tokenizer.pad_token_id).long()

    max_new_tokens = 500

    ## 1) MISTRAL BASELINE
    print("MISTRAL BASELINE")
    mistral = MistralForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.1",
        attn_implementation=config.attn_implementation,
        torch_dtype=config.torch_dtype,
        temperature=data_args.temp,
        do_sample=do_sample,
        use_cache=config.use_cache,
        device_map=config.device_map,
    )

    mistral.eval()
    mistral.to(device)

    output_ids = mistral.generate(
        input_ids=tokenizer_out.to(device),
        attention_mask=attention_mask.to(device),
        max_new_tokens=max_new_tokens,
        output_scores=True,
        return_dict_in_generate=False,
    )
    outputs = tokenizer.batch_decode(
        output_ids, spaces_between_special_tokens=False, skip_special_tokens=True
    )
    print(outputs[0])

    mistral.to(torch.device("cpu"))

    ## 2) UMLS TUNED
    print("\n\nUMLS TUNED")

    model.eval()
    model.to(device)

    prefix = torch.zeros((1, 256)).to(device)

    output_ids = model.generate(
        task=None,
        prefix=prefix,
        input_ids=tokenizer_out.to(device),
        attention_mask=attention_mask.to(device),
        max_new_tokens=max_new_tokens,
        output_scores=True,
        return_dict_in_generate=False,
        debug=False,
    )
    outputs = tokenizer.batch_decode(
        output_ids, spaces_between_special_tokens=False, skip_special_tokens=True
    )
    print(outputs[0])

    model.to(torch.device("cpu"))

    import pdb

    pdb.set_trace()
