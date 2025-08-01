import random
import time
import numpy as np
import json
from collections import defaultdict
import re
import requests
import h5py
from together import Together

# Parameter Settings
datasets_name = "banking77"
datasets_type = "small"
c = 1
epsilon1 = 0.1
epsilon2 = 0.1
llm_api_key = "your api key"


data_path = f'code/datasets/{datasets_name}/{datasets_type}.jsonl'  # Dataset path
prompt_file_path = f'code/constraint_generation/prompts/{datasets_name}_{datasets_type}_ml_prompt' # Prompt prefix path
file_path = f'code/datasets/embeddings/{datasets_name}/{datasets_type}_embeds.hdf5'      # Dataset embedding path
result_ml_path = f"code/constraint_generation/Mls_Result/{datasets_type}/MLS.json" #output ml constraints
query_num_path = f"code/constraint_generation/Mls_Result/{datasets_type}/query_num.txt" #output query times
cluster_k_path = 'code/clustering/utils/cluster_k.json'

with open(cluster_k_path, 'r') as json_file:
    cluster_k = json.load(json_file)
k = cluster_k[f"{datasets_name}"]

if datasets_name == 'tweet':
    input_label = "text"
else:
    input_label = "input"


print("k",k)
print("c",c)
print("epsilon1",epsilon1)
print("epsilon2",epsilon2)
print("data_path",data_path)
print("file_path",file_path)
print("ml_prompt_path",prompt_file_path)
print("result_ml_path",result_ml_path)
print("query_num_path",query_num_path)


# Function
# Load data
def load_json(path):
    data = []
    with open(path, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

# Calculate Euclidean distance
def euclidean_distance_cost(vec1,vec2):
    return np.sum((vec1-vec2)**2)

def euclidean_distance(vec1,vec2):
    return np.sqrt(np.sum((vec1-vec2)**2))

# Min-max initialization
def min_max(data_dict, k):
    # Randomly select a point as the starting point
    selected_points = [random.choice(list(data_dict.keys()))]

    while len(selected_points) < k:
        max_distance = -1
        next_point = None

        # Traverse all points to find the point farthest from the current selected point set
        for key, vector in data_dict.items():
            if key in selected_points:
                continue

            min_distance = min(
                euclidean_distance(np.array(vector), np.array(data_dict[sel_key]))
                for sel_key in selected_points
            )
            if min_distance > max_distance:
                max_distance = min_distance
                next_point = key
        selected_points.append(next_point)
    return selected_points

# Cost calculation
def cost_c(embeddings,target_sequence):
    target_vectors = [embeddings[str(i)] for i in target_sequence]
    total_min_distance = 0
    for key, vector in embeddings.items():
        min_distance = float('inf')
        for target_vector in target_vectors:
            distance = euclidean_distance_cost(vector, target_vector)
            min_distance = min(min_distance, distance)

        # print(min_distance)
        total_min_distance += min_distance
    return total_min_distance

#r_j calculation
def r_j(epsilon1_,j,R,epsilon2_,d):
    return epsilon1_*(np.power(1+epsilon2_,j)*R)/(np.sqrt(10*d))

# Grid partitioning
def search_points_in_grid(data,center, l, r):
    """
    Search for data points in a grid centered on the target point
    :param data: Dataset, in dictionary form {point_id: point_coordinates}
    :param target_point: Target point, in list or array form
    :param l: Extension distance for each dimension of the grid
    :param r: Side length of the small grid
    :return: Points falling within the grid and their grid coordinates
    """

    target_point_embedding = center

    # Calculate the boundaries of the large grid
    lower_bound = target_point_embedding - l  
    upper_bound = target_point_embedding + l 

    points_in_grid = {}

    # Traverse the dataset to determine if points are within the grid
    for point_id, coordinates in data.items():
        large_grid_number = 0
        if np.all(coordinates >= lower_bound) and np.all(coordinates <= upper_bound):
            large_grid_number = large_grid_number + 1
            # Calculate the index of the small grid
            grid_index = tuple(((coordinates - lower_bound) // r).astype(int))
            points_in_grid[point_id] = grid_index
    return points_in_grid

# Calculate the maximum distance in a list
def compute_max_distance(lst, embeddings):
    max_distance = 0
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            v1 = embeddings[lst[i]]
            v2 = embeddings[lst[j]]
            distance = euclidean_distance(v1, v2)
            max_distance = max(max_distance, distance)
    return max_distance

# Load ml_bank_prompt
def load_prompt(prompt_file_path):
    prompt = ''
    with open(prompt_file_path, 'r') as file:
        for line in file:
            l = line.strip() + ' '
            prompt += l
    return prompt


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
            error_msg = str(e)
            if 'Error code: 422' in error_msg:
                return '422'
            retries += 1
            print(f"API call failed ({retries}/{max_retries}), error: {e}, waiting {delay} seconds before retrying...")
            time.sleep(delay)
            delay += 2  # Increase wait time by 2 for each retry
            delay += random.uniform(0, 1)  # Add some randomness to avoid synchronized retries

    raise Exception(f"API call failed, maximum retries {max_retries} reached")

# Main
# Load data
data = load_json(data_path)
data_to_dict = {}
for num,dt in enumerate(data):
    data_to_dict[num] = dt
print('----Data loaded successfully----')

# Open HDF5 file
with h5py.File(file_path, 'r') as file:
    if 'embeds' in file:
        dataset = file['embeds'][:]
embedding_dict = {}
for i in range(len(dataset)):
    embedding_dict[str(i)] = np.asarray(dataset[i])
print('----Embeddings loaded successfully----')

n = len(embedding_dict)
print('n size:', n)

d = len(embedding_dict['0'])
print('n size:', n)

# Randomly select point p
random.seed(7)
p_key = random.choice(list(embedding_dict.keys()))
p = np.array(embedding_dict[p_key])
print('Random point:', p_key)


# Calculate the distance between p and other points, and find the farthest point P'
max_distance = -1
p_prime_key = None
for key, value in embedding_dict.items():
    if key != p_key:
        distance = euclidean_distance(p, np.array(value)) 
        if distance > max_distance:
            max_distance = distance
            p_prime_key = key

# Get the farthest point P''
p_prime = np.array(embedding_dict[p_prime_key])

# Calculate the distance T between p and P'
T = max_distance
print(f"Distance T between p and P': {T}")


# Use min-max to select k center points
initialize_points = min_max(embedding_dict, k)
print('Initial points:', initialize_points, '===Length:', len(initialize_points))

# Calculate costs
costs = cost_c(embedding_dict,initialize_points)
print('cost:',costs)

# Calculate R
R = np.sqrt(costs/(c*n))
print('R size:', R)

Lenr_j = []
j = 0
while r_j(epsilon1,j,R,epsilon2,d) < T:
    Lenr_j.append(r_j(epsilon1,j,R,epsilon2,d))
    j = j + 1

SPIG_LIST  = []# Store marked grids
for j in range(len(Lenr_j)):
    r = Lenr_j[j]

    SPIG = search_points_in_grid(embedding_dict,p,T,r)
    SPIG_LIST.append(SPIG)

print('Number of large grids:', len(SPIG_LIST))

# Find points with the same grid coordinates in i
# Prepare ML query labels
res_dict = {}
for i in range(len(SPIG_LIST)):
    data_dict = SPIG_LIST[i]
    coord_dict = defaultdict(list)

    # Group keys in the data dictionary by coordinates
    for key, coord in data_dict.items():
        coord_dict[coord].append(key)

    # Filter out coordinate groups with more than 2 and less than 100 keys
    result = [keys for keys in coord_dict.values() if len(keys) >= 2 and len(keys)<=100]
    res_dict[i] = result

# Sort
for k in range(len(res_dict)):
    ml_query = []
    for i in res_dict[k]:
        if set(i) not in ml_query:
            ml_query.append(set(i))
    ml_query = [list(i) for i in ml_query]

    distances = []
    for lst in ml_query:
        max_dist = compute_max_distance(lst,embedding_dict)
        distances.append((lst, max_dist))

    distances.sort(key=lambda x: x[1])

    sorted_ml_query = [lst for lst, dist in distances]
    res_dict[k] = sorted_ml_query

# Remove duplicates
repeat_set = []
for num,ml_query in res_dict.items():
    ml_query_copy  = [ml for ml in ml_query]
    for i in ml_query:
        if set(i) not in repeat_set:
            repeat_set.append(set(i))
        else:
            ml_query_copy.remove(i)
    res_dict[num] = ml_query_copy
print(res_dict)

# Load ml_bank_prompt
ml_tag_coding = []       # Record which points are marked
Record_fusion_dict ={}   # Record the dictionary of fused points
kpi_num = 0              # Record the number of queries
Different_proportion_of_query_times = {}
get_proportion_or_not = {'2':False,'4':False,'6':False,'8':False,'10':False,'20':False,'30':False,'40':False}
for num,ml_query in res_dict.items():
    if len(ml_query) > 0:
        Record_fusion_dict[str(num)] = {}
        #get_ml
        for test1 in ml_query:
            try:
                if len(ml_tag_coding) >= (n*0.02) and get_proportion_or_not["2"] == False:
                    Different_proportion_of_query_times["2"] =  kpi_num
                    get_proportion_or_not["2"] = True

                if len(ml_tag_coding) >= (n*0.04) and get_proportion_or_not["4"] == False:
                    Different_proportion_of_query_times["4"] =  kpi_num
                    get_proportion_or_not["4"] = True

                if len(ml_tag_coding) >= (n*0.06) and get_proportion_or_not["6"] == False:
                    Different_proportion_of_query_times["6"] = kpi_num
                    get_proportion_or_not["6"] = True

                if len(ml_tag_coding) >= (n*0.08) and get_proportion_or_not["8"] == False:
                    Different_proportion_of_query_times["8"] = kpi_num
                    get_proportion_or_not["8"] = True

                if len(ml_tag_coding) >= (n*0.1) and get_proportion_or_not["10"] == False:
                    Different_proportion_of_query_times["10"] = kpi_num
                    get_proportion_or_not["10"] = True

                if len(ml_tag_coding) >= (n*0.20) and get_proportion_or_not["20"] == False:
                    Different_proportion_of_query_times["20"] = kpi_num
                    get_proportion_or_not["20"] = True

                if len(ml_tag_coding) >= (n*0.3) and get_proportion_or_not["30"] == False:
                    Different_proportion_of_query_times["30"] = kpi_num
                    get_proportion_or_not["30"] = True

                if len(ml_tag_coding) >= (n*0.4):
                    print('40% of points are marked')
                    Different_proportion_of_query_times["40"] = kpi_num
                    break

                test1 = [i for i in test1 if i not in ml_tag_coding]
                if len(test1) > 1: # After cleaning, ensure ML has at least two points
                    print(f'Obtained ml_query: {test1}')

                    ml_bank_prompt = load_prompt(prompt_file_path)
                    for i in range(len(test1)):
                        ml_bank_prompt = ml_bank_prompt + ' Target text ' + str(i+1) + ': ' + data_to_dict[int(test1[i])][input_label]

                    # Query LLM
                    res = call_deepseek_api_with_backoff(ml_bank_prompt)

                    if res == '422':
                        continue

                    kpi_num += 1
                    print('res:', res)
                    matches = re.findall(r'\[(.*?)\]', res)

                    if matches:
                        lists = [list(map(int, match.split(','))) for match in matches]
                        for ml in lists:
                            if len(ml) >= 2:
                                print('------Obtained ML:', [test1[ml[i] - 1] for i in range(len(ml))])

                                demo = []
                                for j in [test1[ml[i]-1] for i in range(len(ml))]:
                                    demo.append(data_to_dict[int(j)]['label'])
                                print('Fused ML labels:', demo)

                                ## Assign an ID and record the fusion and update ML
                                ID = n+int(test1[ml[0]-1])
                                print("New ID:", ID)

                                Mapping_list = [test1[ml[i]-1] for i in range(len(ml))]

                                Record_fusion_dict[str(num)][str(ID)] = Mapping_list
                                print(Record_fusion_dict)

                                for i in Mapping_list:
                                    ml_tag_coding.append(str(i))
            except Exception as e:
                # When an error is caught, print the error message and skip the current iteration
                print(f"Error message: {e}")
                continue
    if len(ml_tag_coding) >= (n*0.4):
        print('40% of points are marked')
        break

print("ML marking completed")
print(f'Number of queries: {kpi_num}')
print(Record_fusion_dict)


# Save ML
with open(result_ml_path, "w", encoding="utf-8") as json_file:
    json.dump(Record_fusion_dict, json_file, ensure_ascii=False, indent=4)

print(Different_proportion_of_query_times)
# Save query count
with open(query_num_path, "w", encoding="utf-8") as file:
    file.write(f"Total number of queries,{kpi_num}\n")
    file.write(str(Different_proportion_of_query_times))
