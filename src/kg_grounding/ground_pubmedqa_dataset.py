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
    dsl = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
    dsl = dsl["train"]

    shuffled_dataset = dsl.shuffle(seed=42)
    subset_train = shuffled_dataset.select(range(500))  # First 500 samples
    subset_test = shuffled_dataset.select(range(500, 1000))  # Next 500 samples

    # Initialize lists for each subset
    subset_train_dicts = []
    subset_test_dicts = []

    codes = {"yes": "A)", "no": "B)", "maybe": "C)"}

    # Define a function to create the dictionary format for each subset
    def create_subset_dict(subset):
        result = []
        ans = []
        for entry in subset:
            pubid = entry["pubid"]
            question = entry["question"]
            context = " ".join(entry["context"]["contexts"])
            final_decision = (
                codes[entry["final_decision"]] + " " + entry["final_decision"]
            )

            # Create the dictionary in the specified format
            formatted_entry = {
                pubid: [
                    {
                        "role": "user",
                        "content": f"Please address the following medical question based on the Context and any useful information you may find in the given concepts from a medical graph.\nQuestion: {question}\nContext: {context} Options: A) yes\nB) no\nC) maybe\nAnswer with the best option directly. Ignore irrelevant information. Graph: {{kg_embedding}}",
                    },
                    {"role": "assistant", "content": final_decision},
                ]
            }
            result.append(formatted_entry)
            ans.append(entry["final_decision"])
        print(len(ans), Counter(ans))
        return result

    # Apply the function to each subset
    subset_train_dicts = create_subset_dict(subset_train)
    subset_test_dicts = create_subset_dict(subset_test)
    # 500 Counter({'yes': 276, 'no': 173, 'maybe': 51})
    # 500 Counter({'yes': 276, 'no': 165, 'maybe': 59})

    # print()
    # print(subset_train_dicts[0])  # Check the format for subset 1
    # print()

    # Save each subset to a separate JSONL file
    save_to_jsonl(
        subset_train_dicts,
        f"{ROOT}/pubmedqa/train_with_graph_embeds_no_marker_end.jsonl",
    )
    save_to_jsonl(
        subset_test_dicts,
        f"{ROOT}/pubmedqa/test_with_graph_embeds_no_marker_end.jsonl",
    )

    print("Subsets saved to JSONL files!")
    print(
        f"Train at {ROOT}/pubmedqa/train_with_graph_embeds_no_marker_end.jsonl"
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
    # Use regular expression to capture text between '\nQuestion:' and 'Options:'
    match = re.search(r"\nQuestion:(.*?) Options:", text, re.DOTALL)
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
    # print(data)

    print(f"grounded concepts saved to {output_path}")


if __name__ == "__main__":
    """You can run this script off-line, locally."""
    
    ### 1. Create JSONL
    create_jsonl_files()

    ### 2. Create Grounding files 
    umls_vocab_path = f"{ROOT}/umls/vocab.csv"

    # TRAIN
    data_path = f"{ROOT}/pubmedqa/train_with_graph_embeds_no_marker_end.jsonl"
    output_path = f"{ROOT}/pubmedqa/train_grounding.jsonl"
    create_grounding(data_path, umls_vocab_path, output_path)

    # TEST
    data_path = f"{ROOT}/pubmedqa/test_with_graph_embeds_no_marker_end.jsonl"
    output_path = f"{ROOT}/pubmedqa/test_grounding.jsonl"
    create_grounding(data_path, umls_vocab_path, output_path)
