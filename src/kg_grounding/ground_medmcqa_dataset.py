import re
import json
import spacy

from datasets import load_dataset
from collections import Counter
from tqdm import tqdm
from utils import load_jsonl_file
from scispacy.linking import EntityLinker

nlp = None
linker = None
ROOT = "/placeholder/to/your/data/directory"


def save_to_jsonl(data, filename):
    with open(filename, "w") as f:
        for entry in data:
            json.dump(entry, f)
            f.write("\n")  # Add a newline after each JSON object


def create_jsonl_files():
    dsl = load_dataset("openlifescienceai/medmcqa")
    subset_train = dsl["train"]
    # Due to the unavailability of answer keys for the test set, 
    # we follow others \cite{wu2023pmcllamabuildingopensourcelanguage,
    # tu2023generalistbiomedicalai,labrak-etal-2024-biomistral} 
    # and report results on the validation set.
    subset_test = dsl["validation"]

    # Initialize lists for each subset
    subset_train_dicts = []
    # subset_dev_dicts = []
    subset_test_dicts = []

    # Define a function to create the dictionary format for each subset
    def create_subset_dict(subset, train_with_exp=False):
        result = []
        ans = []
        for entry in subset:
            pubid = entry["id"]
            question = entry["question"]
            response = entry["cop"]
            choices = f"A) {entry['opa']}\nB) {entry['opb']}\n C) {entry['opc']}\n D) {entry['opd']}"
            code2ans = {
                0: f"A) {entry['opa']}", 
                1: f"B) {entry['opb']}", 
                2: f"C) {entry['opc']}",
                3: f"D) {entry['opd']}"
            }

            # Create the dictionary in the specified format
            if train_with_exp:
                explanation = entry["exp"]
                formatted_entry = {
                    pubid: [
                        {
                            "role": "user",
                            "content": f"You're the best doctor. Please address the following medical question based on any useful information you may find in the given concepts from a medical graph.\nQuestion: {question}\nOptions: {choices}\Explain your answer. Ignore irrelevant information. Graph: {{kg_embedding}}",
                        },
                        {"role": "assistant", "content": f"{explanation}. Therefore, the correct answer is {code2ans[response]}"},
                    ]
                }
            else:
                formatted_entry = {
                    pubid: [
                        {
                            "role": "user",
                            "content": f"You're the best doctor. Please address the following medical question based on any useful information you may find in the given concepts from a medical graph.\nQuestion: {question}\nOptions: {choices}\nAnswer with the best option directly. Ignore irrelevant information. Graph: {{kg_embedding}}",
                        },
                        {"role": "assistant", "content": code2ans[response]},
                    ]
                }

            result.append(formatted_entry)
            ans.append(response)
        print(len(ans), Counter(ans))
        return result

    # Apply the function to each subset
    print("Create train file")
    subset_train_dicts = create_subset_dict(subset_train)
    print("Create train file with explanation")
    subset_train_explain_dicts = create_subset_dict(subset_train, True)
    print("Create test file")
    subset_test_dicts = create_subset_dict(subset_test)

    # print()
    # print(subset_train_dicts[0])  # Check the format for subset 1
    # print()

    # Save each subset to a separate JSONL file
    save_to_jsonl(
        subset_train_dicts,
        f"{ROOT}/medmcqa/train_with_graph_embeds_no_marker_end.jsonl",
    )
    save_to_jsonl(
        subset_train_explain_dicts,
        f"{ROOT}/medmcqa/train_with_explanation_and_graph_embeds_no_marker_end.jsonl",
    )
    save_to_jsonl(
        subset_test_dicts,
        f"{ROOT}/medmcqa/test_with_graph_embeds_no_marker_end.jsonl",
    )

    print("Subsets saved to JSONL files!")
    print(
        f"Train at {ROOT}/medmcqa/train_with_graph_embeds_no_marker_end.jsonl"
    )


def load_umls_vocab(vocab_path):
    with open(vocab_path, "r", encoding="utf8") as fin:
        data = fin.readlines()
        cui_vocab = [line.split("\t")[0] for line in data]
    return cui_vocab


def load_entity_linker(threshold=0.90):
    nlp = spacy.load("en_core_sci_sm")
    # linker = EntityLinker(resolve_abbreviations=True, name="umls", threshold=threshold)
    # nlp.add_pipe(linker)
    nlp.add_pipe(
        "scispacy_linker",
        config={
            "resolve_abbreviations": True,
            "linker_name": "umls",
            "threshold": threshold,
        },
    )
    linker = nlp.get_pipe("scispacy_linker")
    return nlp, linker


def entity_linking_to_umls(sentence, nlp, linker):
    doc = nlp(sentence)
    entities = doc.ents
    all_entities_results = []
    for mm in range(len(entities)):
        entity_text = entities[mm].text
        entity_start = entities[mm].start
        entity_end = entities[mm].end
        all_linked_entities = entities[mm]._.kb_ents
        all_entity_results = []
        for ii in range(len(all_linked_entities)):
            curr_concept_id = all_linked_entities[ii][0]
            curr_score = all_linked_entities[ii][1]
            curr_scispacy_entity = linker.kb.cui_to_entity[curr_concept_id]
            curr_canonical_name = curr_scispacy_entity.canonical_name
            curr_TUIs = curr_scispacy_entity.types
            curr_entity_result = {
                "Canonical Name": curr_canonical_name,
                "Concept ID": curr_concept_id,
                "TUIs": curr_TUIs,
                "Score": curr_score,
            }
            all_entity_results.append(curr_entity_result)
        curr_entities_result = {
            "text": entity_text,
            "start": entity_start,
            "end": entity_end,
            "start_char": entities[mm].start_char,
            "end_char": entities[mm].end_char,
            "linking_results": all_entity_results,
        }
        all_entities_results.append(curr_entities_result)
    return all_entities_results


def ground_mentioned_concepts(nlp, linker, sent, umls_vocab):
    ent_link_results = entity_linking_to_umls(sent, nlp, linker)
    mentioned_concepts = set()
    for ent_obj in ent_link_results:
        for ent_cand in ent_obj["linking_results"]:
            CUI = ent_cand["Concept ID"]
            if CUI in umls_vocab:
                mentioned_concepts.add(CUI)
    return mentioned_concepts


def ground_context(id_, s, umls_vocab):
    global nlp, linker
    if nlp is None or linker is None:
        print("Loading scispacy...")
        nlp, linker = load_entity_linker()
        print("Loaded scispacy.")

    # id_, s, umls_vocab = item
    question_concepts = ground_mentioned_concepts(nlp, linker, s, umls_vocab)
    if len(question_concepts) == 0:
        print(f"for {s}, concept not found in umls linking.")

    # question_concepts = question_concepts -  answer_concepts
    question_concepts = sorted(list(question_concepts))
    return {id_: {"qc": question_concepts}}


def match_mentioned_concepts(ids, sents, umls_vocab):
    res = []
    for ids_i, sents_i in tqdm(zip(ids, sents), total=len(ids)):
        res_i = ground_context(ids_i, sents_i, umls_vocab)
        res.append(res_i)
    return res


def extract_content(text):
    # Use regular expression to capture text between '\nQuestion:' and '\nOptions:'
    match = re.search(r"\nQuestion:(.*?)\nOptions:", text, re.DOTALL)
    if match:
        return match.group(1).strip()  # Remove any leading/trailing whitespace
    return None  # Return None if no match is found


def ground_umls(data_path, umls_vocab_path):
    print(f"Loading UMLS vocab from {umls_vocab_path}")
    umls_vocab = set(load_umls_vocab(umls_vocab_path))
    print(f"{len(umls_vocab)} CUIs loaded. E.g.: {list(umls_vocab)[:3]}")

    ids = []
    sents = []
    data = load_jsonl_file(data_path, "dict")

    # i = 0
    for s_id, content in tqdm(data.items()):
        ids.append(s_id)
        prompt = extract_content(content[0]["content"])
        sents.append(prompt)
        # i += 1
        # if i == 5:
        #     break

    res = match_mentioned_concepts(ids, sents, umls_vocab)
    return res


def create_grounding(statement_path, umls_vocab_path, output_path):
    print("create_grounding")
    data = ground_umls(statement_path, umls_vocab_path)

    save_to_jsonl(data, output_path)

    print(f"grounded concepts saved to {output_path}")


if __name__ == "__main__":
    """You can run this script off-line, locally."""

    ### 1. Create JSONL
    create_jsonl_files()
    # Create train file
    # 182822 Counter({0: 53591, 1: 47826, 2: 42442, 3: 38963})
    # Create train file with explanation
    # 182822 Counter({0: 53591, 1: 47826, 2: 42442, 3: 38963})
    # Create test file
    # 4183 Counter({0: 1348, 1: 1085, 2: 925, 3: 825})

    ### 2. Create Grounding files 
    umls_vocab_path = f"{ROOT}/umls/vocab.csv"

    # TRAIN
    data_path = f"{ROOT}/medmcqa/train_with_graph_embeds_no_marker_end.jsonl"
    output_path = f"{ROOT}/medmcqa/train_grounding.jsonl"
    create_grounding(data_path, umls_vocab_path, output_path)

    # TEST
    data_path = f"{ROOT}/medmcqa/test_with_graph_embeds_no_marker_end.jsonl"
    output_path = f"{ROOT}/medmcqa/test_grounding.jsonl"
    create_grounding(data_path, umls_vocab_path, output_path)