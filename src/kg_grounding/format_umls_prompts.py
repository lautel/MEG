import sys
import json
import random
import multiprocessing
import pandas as pd
from tqdm import tqdm
from collections import Counter
from sklearn.utils import shuffle

# Format 1)
def prompt_base(basedir):
    input_file_path = f'{basedir}/umls/embeddings/vocab.csv'
    output_file_path = f'{basedir}/umls/embeddings/vocab_training_meg.jsonl'
    df=pd.read_csv(input_file_path, sep="\t", names=["code", "name"])
    print(f"{len(df)} lines loaded")
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for idx, data in df.iterrows():
            code = data['code']
            entity = data['name']
            input_content = "Explain to me this medical concept {kg_embedding}"
            assistant_content = f"{entity}"
            output_row = {idx: [
                {"role": "user", "content": input_content},
                {"role": "assistant", "content": assistant_content}
            ]}
            output_file.write(json.dumps(output_row) + '\n')
        print(f"{output_file_path} written")


# Format 2)
def prompt_triplets(basedir):
    input_file_path = f'{basedir}/umls/embeddings/triplets.jsonl'
    output_file_path = f'{basedir}/umls/embeddings/vocab_triplets_training.jsonl'
    df=pd.read_json(input_file_path, lines=True)
    print(f"{len(df)} lines loaded")

    # Run multiprocessing 
    num_processes = 12
    pool = multiprocessing.Pool(num_processes)
    cpus=multiprocessing.cpu_count()
    # calculate the chunk size as an integer
    chunk_size = int(len(df) / num_processes)
    chunks = [
        df.iloc[df.index[i : i + chunk_size]] for i in range(0, df.shape[0], chunk_size)
    ]
    print(f"{num_processes} processes, {len(chunks)} chunks of size {chunk_size}. Node with {cpus} CPUs")
    
    # run
    result = pool.map(calc_stuff, chunks)
    # Flatten list 
    results = [item for sublist in result for item in sublist]
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for output_row in results:
            output_file.write(json.dumps(output_row) + '\n')
    print(f"{output_file_path} written")

def calc_stuff(chunk_df):
    output = []
    chunk_df["verbal"] = chunk_df.apply(lambda row: 
                                        row["a_text"] + " " + row["rel"].replace("_"," ") + " " + row["b_text"] 
                                        , axis=1)
    for idx, data in tqdm(chunk_df.iterrows(), total=len(chunk_df), leave=False):
        if data["a_text"] == data["b_text"]:
            continue
        input_content = data["verbal"]
        assistant_content = ""
        output_row = {idx: {"triplet": input_content, "entity": data["a_text"], "code": data["a"]}}
        output.append(output_row)
    return output


# Format 3) -- include more than 1 concept in the same instance
def prompt_extended(basedir):
    input_file_path = f'{basedir}/umls/embeddings/vocab.csv'
    output_file_path = f'{basedir}/umls/embeddings/vocab_training_meg_extended.jsonl'
    df=pd.read_csv(input_file_path, sep="\t", names=["code", "name"])
    df.dropna(inplace=True)
    df.drop(columns=["code"], inplace=True)

    # Define the minimum and maximum group size for names to combine
    min_names_per_row = 2
    max_names_per_row = 10  # You can adjust this depending on how many names you want per group

    combined_names, group_sizes = [], []
    while len(group_sizes) < len(df):
        i = 0
        while i < len(df):
            group_size = random.randint(min_names_per_row, max_names_per_row)
            name_group = df['name'].iloc[i:i+group_size]
            combined_name = '; '.join(name_group)
            combined_names.append(combined_name)
            i += group_size
            group_sizes.append(group_size)
        print(f"{len(group_sizes)}/{len(df)}")

    combined_df1 = pd.DataFrame({'name': combined_names})
    print(dict(Counter(group_sizes)))

    # Merge and shuffle    
    merged_df = pd.concat([df, combined_df1], ignore_index=True)
    shuffled_df = shuffle(merged_df).reset_index(drop=True)

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for idx, data in shuffled_df.iterrows():
            entities = data['name']
            input_content = "Explain to me these medical concepts {kg_embedding}"
            assistant_content = f"{entities}"
            output_row = {idx: [
                {"role": "user", "content": input_content},
                {"role": "assistant", "content": assistant_content}
            ]}
            output_file.write(json.dumps(output_row) + '\n')
    print(f"{output_file_path} written")
    print(f"{len(shuffled_df)} lines") # 645384 lines

if __name__ == "main":
    basedir = "/placeholder/to/your/data/directory"
    mode = str(sys.argv[1])
    assert mode in ["base", "triplets", "ext"]

    prompts = {
        "basic": prompt_base,
        "triplets": prompt_triplets,
        "ext": prompt_extended
    }

    prompts[mode](basedir)