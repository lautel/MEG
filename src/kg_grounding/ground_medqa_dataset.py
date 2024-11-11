import json
import pandas as pd

ROOT = "/placeholder/to/your/data/directory"

### Extract ground-truth grounding 
for split in ["train", "dev", "test"]:
    print(f"Processing split: {split}")
    input_file_path = f'{ROOT}/medqa_usmle/statement/{split}.statement.umls_linked.jsonl'
    input_file_path_gr = f'{ROOT}/medqa_usmle/grounding/{split}.grounded.jsonl'
    output_file_path = f'{ROOT}/medqa_usmle/{split}.jsonl'
    output_file_path_gr = f'{ROOT}/medqa_usmle/{split}_grounding.jsonl'

    grounded = pd.read_json(input_file_path_gr, lines=True)
    grounded['sent'] = grounded.apply(lambda row: row['sent'][:-len(row['ans'])], axis=1)
    grounded['sent'] = grounded['sent'].str.strip()
    grounded = grounded.drop(['ac', 'ans'], axis=1)
    grounded['qc'] = grounded['qc'].apply(tuple)
    grounded = grounded.drop_duplicates(subset=['sent', 'qc']).reset_index(drop=True)
    grounded['qc'] = grounded['qc'].apply(list)

    with open(input_file_path, 'r', encoding='utf-8') as input_file, open(output_file_path, 'w', encoding='utf-8') as output_file, open(output_file_path_gr, 'w', encoding='utf-8') as output_file_gr:
        for line in input_file:
            data = json.loads(line)
            entry_id = data['id']
            question_text = data['question']['stem']
            options = '\n'.join([f"{choice['label']}) {choice['text']}" for choice in data['question']['choices']])
            input_content = f"Please, address the following medical question based in the Input text. Answer with the best option directly.\nInput: {question_text} Options: {options}"
            correct_answer_label = data['answerKey']
            correct_answer_text = [choice['text'] for choice in data['question']['choices'] if choice['label'] == correct_answer_label][0]
            assistant_content = f"{correct_answer_label}) {correct_answer_text}"
            output_row = {entry_id: [
                {"role": "user", "content": input_content},
                {"role": "assistant", "content": assistant_content}
            ]}
            output_file.write(json.dumps(output_row) + '\n')

            concept_ids = []
            for concepts in grounded.loc[grounded['sent'] == question_text, 'qc'].tolist():
                concept_ids.extend(concepts)
            concept_ids = list(set(concept_ids))

            concept_ids_ans = []
            for entity in data['question']['choices']:
                for result in entity['text_ents']:
                    for linking_result in result['linking_results']:
                        concept_ids_ans.append(linking_result['Concept ID'])
            output_row_gr = {entry_id: {'qc': concept_ids, 'ac': concept_ids_ans}}
            output_file_gr.write(json.dumps(output_row_gr) + '\n')
        print(f"{output_file_path} written")
        print(f"{output_file_path_gr} written\n")


### Format prompt to include graph embeddings
for split in ["train", "dev", "test"]:
    print(f"Processing split: {split}")
    input_file_path = f'{ROOT}/medqa_usmle/statement/{split}.statement.umls_linked.jsonl'
    output_file_path = f'{ROOT}/medqa_usmle/{split}_with_graph_embeds.jsonl'

    with open(input_file_path, 'r', encoding='utf-8') as input_file, open(output_file_path, 'w', encoding='utf-8') as output_file:
        for line in input_file:
            data = json.loads(line)
            entry_id = data['id']
            question_text = data['question']['stem']
            options = '\n'.join([f"{choice['label']}) {choice['text']}" for choice in data['question']['choices']])
            
            # input_content is the only difference with the code above 
            input_content = f"Please address the following medical question based on the Input text and any useful information you may find in the given medical concepts {{kg_embedding}} \nAnswer with the best option directly. Ignore irrelevant information.\nInput: {question_text} Options: {options}"
            
            correct_answer_label = data['answerKey']
            correct_answer_text = [choice['text'] for choice in data['question']['choices'] if choice['label'] == correct_answer_label][0]
            assistant_content = f"{correct_answer_label}) {correct_answer_text}"
            output_row = {entry_id: [
                {"role": "user", "content": input_content},
                {"role": "assistant", "content": assistant_content}
            ]}
            output_file.write(json.dumps(output_row) + '\n')

        print(f"{output_file_path} written")
