# MEG
We provide the source code and data of our paper [MEG: Medical Knowledge-Augmented Large Language Models for Question Answering](https://arxiv.org/abs/2411.03883)

<p align="center">
  <img src="./assets/overview.png" width="500" title="MEG overview" alt="">
</p>

## 1\. Installation
Create a fresh conda environment and install all dependencies. 
You can do so from the YAML files provided. 
```text
conda env create --file=conda_env_meg-mistral.yml --name=meg
```
and/or
```text
conda env create --file=conda_env_meg-llama.yml --name=meg2
```

Note 1: Both environments install **Python 3.11.5**; *conda_env_meg-mistral.yml* installs `pytorch-cuda=11.8` and *conda_env_meg-llama.yml* installs `pytorch-cuda=12.1`.
Note 2: MEG-Mistral models can be run with either **deepspeed** or **torchrun**. MEG-Llama models are run with **torchrun**.

#### 1.1\. Installation notes

* Make sure your CUDA version is compatible with the version that torch was compiled with. Visiting [Get Started @PyTorch](https://pytorch.org/get-started/locally/) may be helpful.  
* If you run into an `undefined symbol` error using Flash Attention, you may want to visit this [issue](https://github.com/Dao-AILab/flash-attention/issues/620).
* If you run into `no module named transformers.cache_utils`, this [discussion](https://huggingface.co/DiscoResearch/mixtral-7b-8expert/discussions/9#6576edcd0370e52e3b2c0620) may be helpful.


#### 1.2\. Configuration
The main entry points to the run the code are in the [scripts/](scripts/) folder. You can find the path configuration in [main.config](main.config). Please, edit this file at your own convenience.

Also, you would need to replace the placeholder `your-hugginface-login-token` with your personal Hugging Face token. This can be found right after the imports in the main Python scripts for training and evaluation in [src/](src/). 

## 2\. Data

### 2.1\. UMLS data
You can download the preprocessed data from [here](https://drive.google.com/file/d/14Rx7bEpJW0_AOOIhsSkkrypNuUxvXRAX/view?usp=sharing) (339 MB). This includes the files needed for the Phase I of training (see Section 4.2 in our [paper](https://arxiv.org/pdf/2411.03883)):
* the UMLS vocabulary and the 297,927 corresponding instances to perform instruction tuning on the mapping network and the LLM.
* the trained Knowledge Graph Embeddings (KGE) on UMLS using GraphSAGE.

The resulting file structure should look like this:
```plain
.
└── umls/
      ├── neighbors_data_n25.jsonl
      ├── test_triplets.jsonl
      ├── triplets.jsonl
      ├── vocab_parents.jsonl
      ├── vocab_training_meg_extended.jsonl
      ├── vocab_training_meg.jsonl
      ├── vocab.csv

      └── embeddings/
            ├── word2embed_edgegraphsage_sapbertinit.h5
            └── word2embed_graphsage_sapbertinit.h5
```


### 2.2\. Question Answering (QA) data
We downloaded question-answering datasets from the following sources:

* MedQA-USMLE - We used the preprocessed data distributed by [DRAGON](https://github.com/michiyasunaga/dragon?tab=readme-ov-file#biomedical-domain)
* PubMedQA - [qiaojin/PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA)
* MedMCQA - [openlifescienceai/medmcqa](https://huggingface.co/datasets/openlifescienceai/medmcqa)
* MMLU-Medical -  [cais/mmlu](https://huggingface.co/datasets/cais/mmlu)

### 2.3\. Model checkpoints
Checkpoints cooming soon. Feel free to reach out to me if you don't find the checkpoints available.

## 3\. Train MEG

### 3.1\. Phase I training (UMLS)
You can skip this phase of training by downloading the model checkpoint provided in **2.3. Model checkpoints**.
Otherwise, if you would like to train MEG on UMLS, download the data as detailed above, set the paths and environment variables as needed, and run
```
cd scripts/mistral/
bash train_meg_umls.sh
```

### 3.2\. Phase II training (QA dataset)
To fine-tune MEG on QA datasets, it is required to load MEG's weights from previous step (phase I). Please, update the value of the variable `umls_meg_ckpt` in the bash files accordingly.

To fine-tune MEG-Mistral on MedQA, PubMedQA, MedMCQA, MMLU-Medical, run:
```
cd scripts/mistral/

bash run_medqa_usmle.sh
bash run_pubmedqa.sh
bash run_medmcqa.sh
bash run_eval_mmlu_medical_from_medmcqa.sh
bash run_eval_mmlu_medical_from_medqa.sh
```

Similarly, to fine-tune MEG-Llama on MedQA, PubMedQA, MedMCQA, MMLU-Medical, run:
```
cd scripts/llama/

bash run_medqa_usmle.sh
bash run_pubmedqa.sh
bash run_medmcqa.sh
bash run_eval_mmlu_medical_from_medmcqa.sh
bash run_eval_mmlu_medical_from_medqa.sh
```

## 4\. License 
This work is licensed under the Apache License 2.0 license. See [`LICENSE`](LICENSE) for details. 
Third-party software and data sets are subject to their respective licenses. <br>
If you find our code/data/models or ideas useful in your research, please consider citing the paper:

```
@misc{cabello2024megmedicalknowledgeaugmentedlarge,
      title={MEG: Medical Knowledge-Augmented Large Language Models for Question Answering}, 
      author={Laura Cabello and Carmen Martin-Turrero and Uchenna Akujuobi and Anders Søgaard and Carlos Bobed},
      year={2024},
      eprint={2411.03883},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2411.03883}, 
}
```