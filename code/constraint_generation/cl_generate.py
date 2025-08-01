import json
import random
import numpy as np
import time
import re
import requests
import h5py
from together import Together

# Parameter Settings
datasets_name = "banking77"
datasets_type = "small"
constraint_number = 50
llm_api_key = "your api key"


data_path = f'code/datasets/{datasets_name}/{datasets_type}.jsonl'  # Dataset path
prompt_file_path = f'code/constraint_generation/prompts/{datasets_name}_{datasets_type}_cl_prompt' # Prompt prefix path
file_path = f'code/datasets/embeddings/{datasets_name}/{datasets_type}_embeds.hdf5'      # Dataset embedding path
query_num_file = f'code/constraint_generation/Cls_Result/{datasets_name}_{datasets_type}_cl_query_num.txt' # Query count record file
constrained_output_file = f'code/constraint_generation/Cls_Result/{datasets_type}/CLs_seed' # Constrained output file
cluster_k_path = 'code/clustering/utils/cluster_k.json'

with open(cluster_k_path, 'r') as json_file:
    cluster_k = json.load(json_file)
k = cluster_k[f"{datasets_name}"]

if datasets_name == 'tweet':
    input_label = "text"
else:
    input_label = "input"

print("==k:",k)
print("==data_path:",data_path)
print("==prompt_file_path:",prompt_file_path)
print("==file_path:",file_path)
print("==query_num_file:",query_num_file)
print("==constrained_output_file:",constrained_output_file)

sd_list = [i for i in range(0,constraint_number)] # Random seed list
print("==sd_list:",sd_list)


# Function
# Start Time
start_time = time.time()

# Load JSON File
def load_json(path):
    data = []
    with open(path, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

# Calculate the number of label classes
def get_label_count(data):
    unique_labels = set(item['label'] for item in data)
    label_count = len(unique_labels)
    return label_count

# Calculate Euclidean Distance
def euclidean_distance(vec1,vec2):
    return  np.sqrt(np.sum((vec1-vec2)**2))

def initialize_points(data_dict, k):
    # Randomly select a point as the starting point
    random.seed(7)
    selected_points = [random.choice(list(data_dict.keys()))]
    while len(selected_points) < k:
        max_distance = -1
        next_point = None
        # Traverse all points to find the point farthest from the current selected point set
        for key, vector in data_dict.items():
            if key in selected_points:
                continue
            # Calculate the minimum distance between this point and all points in the set
            min_distance = min(
                np.linalg.norm(np.array(vector) - np.array(data_dict[sel_key]))
                for sel_key in selected_points
            )
            # Update the point with the maximum distance
            if min_distance > max_distance:
                max_distance = min_distance
                next_point = key
        selected_points.append(next_point)
    return selected_points


# Calculate R
def calculate_max_min_distance(data_dict, selected_keys):
    """
    Calculate the minimum Euclidean distance between all points and the initialized set, and return the maximum value R.

    Parameters:
        data_dict (dict): Dictionary where keys are indices and values are vectors.
        selected_keys (list): Keys of the initialized points.

    Returns:
        float: Maximum minimum distance R.
    """
    max_min_distance = -1
    for key, vector in data_dict.items():
        min_distance = min(
            np.linalg.norm(np.array(vector) - np.array(data_dict[sel_key]))
            for sel_key in selected_keys
        )
        if min_distance > max_min_distance:
            max_min_distance = min_distance

    return max_min_distance


# def calculate_euclidean_distances(data_dict, point_list):
#     """
#     Calculate the Euclidean distance between each point in the data dictionary and the specified point list.
#
#     Parameters:
#         data_dict (dict): Dictionary where keys are indices and values are vectors.
#         point_list (list): A list of points (i.e., a set).
#
#     Returns:
#         dict: Minimum distance from each point to the point list, where keys are point indices in the data dictionary and values are distances.
#     """
#     distances = {}
#     for key, vector in data_dict.items():
#         min_distance = min(
#             np.linalg.norm(np.array(vector) - np.array(data_dict[str(target_point)]))
#             for target_point in point_list
#         )
#         distances[key] = min_distance
#     return distances
def initialize_min_distances_np(embeddings_matrix, initial_indices):
    N = embeddings_matrix.shape[0]
    selected_vecs = embeddings_matrix[initial_indices]  # shape: (k, D)
    all_vecs = embeddings_matrix  # shape: (N, D)

    diffs = all_vecs[:, None, :] - selected_vecs[None, :, :]  # shape: (N, k, D)
    dists = np.linalg.norm(diffs, axis=2)  # shape: (N, k)
    min_dists = np.min(dists, axis=1)      # shape: (N,)
    return min_dists


def update_min_distances_np(min_dists, embeddings_matrix, new_index):
    new_vec = embeddings_matrix[new_index]  # shape: (D,)
    all_vecs = embeddings_matrix  # shape: (N, D)

    diffs = all_vecs - new_vec  # shape: (N, D)
    dists = np.linalg.norm(diffs, axis=1)  # shape: (N,)
    np.minimum(min_dists, dists, out=min_dists)


def Llama_70B_instruct_turbo(content):
    client = Together(api_key=llm_api_key)
    response = client.chat.completions.create(
        model = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=[{"role": "user", "content": content}],
    )
    return response.choices[0].message.content

def call_deepseek_api_with_backoff(content, max_retries=999999999, initial_delay=2):
    retries = 0
    delay = initial_delay

    while retries < max_retries:
        try:
            return Llama_70B_instruct_turbo(content)
        except Exception as e:
            retries += 1
            print(f"API call failed ({retries}/{max_retries}),error: {e},waiting {delay} seconds before retrying...")
            time.sleep(delay)
            delay += 2
            delay += random.uniform(0, 1)
    raise Exception(f"API call failed, maximum retries {max_retries} reached")

# Load Prompt
def load_prompt(prompt_file_path):
    prompt = ''
    with open(prompt_file_path, 'r') as file:
        for line in file:
            l = line.strip() + ' '
            prompt += l
    return prompt

def extract_number(text):
    match = re.search(r'\d+', text)
    if match:
        return int(match.group())
    return None

def get_result(CLs, data, name):
    # Save the result
    result = []
    for i in range(len(CLs)):
        for j in range(len(CLs[i])):
            label = data[CLs[i][j]]['label']
            text = data[CLs[i][j]][input_label]
            result.append({'C': i, 'number:': CLs[i][j], 'text:': text, 'label:': label})
    filename = name + '.json'

    with open(filename, 'w') as f:
        for d in result:
            json.dump(d, f)
            f.write('\n')


#Main
data = load_json(data_path)
data_len = len(data)
print('==Data loaded successfully')
print('==Dataset length:', data_len)

k_ = get_label_count(data)
print('==K',k_)

with h5py.File(file_path, 'r') as file:
    keys_name = list(file.keys())[0]
    dataset = file[keys_name][:]

embedding_dict = {}
for i in range(len(dataset)):
    embedding_dict[str(i)] = np.array(dataset[i])
print('==Data embeddings obtained successfully')
embeddings_matrix = np.stack([np.array(embedding_dict[str(i)]) for i in range(len(embedding_dict))])
# shape: (N, D)

# Load prompt
prompt_string = load_prompt(prompt_file_path)
print('==Prompt:', prompt_string)

# Initialize
initialize_list =  initialize_points(embedding_dict,k)
print('==Initial points:', initialize_list, '///Length:', len(initialize_list))

# R
R = calculate_max_min_distance(embedding_dict,initialize_list)
print('R size:', R)

record_points = set()
Different_proportion_of_query_times = {}
query_num = 0
get_proportion_or_not = {'2':False,'4':False,'6':False,'8':False,'10':False,'20':False,'30':False,'40':False}

# Get CLs
for sd in sd_list:
    query_num_seed = 0
    print(f'Random seed: {sd}')
    CLs = []
    MLs = []
    while len(CLs) < 1:
        random.seed(sd)
        n = 1
        cl = []
        c_ = []
        P = random.randint(0, len(data)-1)
        cl.append(P)
        c_.append(P)
        record_points.add(P)
        while len(cl) < k:
            if len(record_points) >= (data_len * 0.02) and get_proportion_or_not["2"] == False:
                Different_proportion_of_query_times["2"] = query_num
                get_proportion_or_not["2"] = True

            if len(record_points) >= (data_len * 0.04) and get_proportion_or_not["4"] == False:
                Different_proportion_of_query_times["4"] = query_num
                get_proportion_or_not["4"] = True

            if len(record_points) >= (data_len * 0.06) and get_proportion_or_not["6"] == False:
                Different_proportion_of_query_times["6"] = query_num
                get_proportion_or_not["6"] = True

            if len(record_points) >= (data_len * 0.08) and get_proportion_or_not["8"] == False:
                Different_proportion_of_query_times["8"] = query_num
                get_proportion_or_not["8"] = True

            if len(record_points) >= (data_len * 0.1) and get_proportion_or_not["10"] == False:
                Different_proportion_of_query_times["10"] = query_num
                get_proportion_or_not["10"] = True

            if len(record_points) >= (data_len * 0.20) and get_proportion_or_not["20"] == False:
                Different_proportion_of_query_times["20"] = query_num
                get_proportion_or_not["20"] = True

            if len(record_points) >= (data_len * 0.3) and get_proportion_or_not["30"] == False:
                Different_proportion_of_query_times["30"] = query_num
                get_proportion_or_not["30"] = True

            if len(record_points) >= (data_len * 0.4) and get_proportion_or_not['40'] == False: 
                Different_proportion_of_query_times["40"] = query_num
                get_proportion_or_not["40"] = True

            # Get the next point
            threshold = R

            if len(c_) == 1:
                min_distances = initialize_min_distances_np(embeddings_matrix, c_)
            else:
                update_min_distances_np(min_distances, embeddings_matrix, c_[-1])

            c_set = set(c_)
            candidates = [i for i in range(len(min_distances)) if min_distances[i] > R and i not in c_set]
            if not candidates:
                print("------------No points satisfy the condition of distance > R----------------")
                break
            # Randomly select a point
            next_P = int(random.choice(candidates))
            print('next_P: ',next_P)
            c_.append(next_P)

            # # Generate prompt
            parts = [prompt_string, f' Query Q: {data[next_P][input_label]}']
            for i, idx in enumerate(cl):
                parts.append(f' target text {i + 1}: {data[idx][input_label]}')
            content = ''.join(parts)

            res = call_deepseek_api_with_backoff(content)
            query_num = query_num + 1
            query_num_seed += 1
            print('///////')
            print(f'{query_num_seed} time res:',res)
            print('///////')
            pattern = r'\b(no|none)\b'
            # Search the text
            match = re.search(pattern, res, re.IGNORECASE)
            if match:
                cl.append(next_P)
                record_points.add(next_P)

        if len(cl) >= 0:
            CLs.append(cl)
            print("==cl:",cl)
            print('Length:', len(cl))
    with open(query_num_file,"a") as f:
        f.write(f"sd_{sd},{query_num_seed}\n")
    get_result(CLs,data,f'{constrained_output_file}{sd}.json')

print('Query times for different proportions:', Different_proportion_of_query_times)
with open(query_num_file,"a") as f:
    f.write(f"Query times for different proportions,{str(Different_proportion_of_query_times)}\n")

# Record end time
end_time = time.time()
# Calculate runtime
elapsed_time = end_time - start_time
print(f"Program runtime: {elapsed_time} seconds")
