import time
import numpy as np
import random
import json
import h5py
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from clustering.utils.kmean_plus_plus import init_cluster_centers
import copy


# Parameters
data_name = 'banking77'
data_type = 'small'             # small,small_v3,small_gt,large
constrain_type = 'cl'           # cl,ml,cl_ml
percentages = [2,4,6,8,10,20,30]# Define the percentages to select
algorithm = "ls"
P = 0.01 # weight

if constrain_type == 'cl':
    cannot_link_path = f'code/clustering/constrains/{data_type}/CLS.json'
    must_link_path = 'None'
if constrain_type == "ml":
    cannot_link_path = 'None'
    must_link_path = f'code/clustering/constrains/{data_type}/MLS.json'
if constrain_type == "cl_ml":
    cannot_link_path = f'code/clustering/constrains/{data_type}/CLS.json'
    must_link_path = f'code/clustering/constrains/{data_type}/MLS.json'

if data_type != "large":
    data_path = f'code/datasets/{data_name}/small.jsonl'
    file_path = f'code/datasets/embeddings/{data_name}/small_embeds.hdf5'
else:
    data_path = f'code/datasets/{data_name}/large.jsonl'
    file_path = f'code/datasets/embeddings/{data_name}/large_embeds.hdf5'

cluster_result_output_path = f"code/clustering/cluster_result_output/{data_name}_{data_type}_{constrain_type}_{algorithm}.csv"

cluster_k_path = 'code/clustering/utils/cluster_k.json'


with open(cluster_k_path, 'r') as json_file:
    cluster_k = json.load(json_file)
K = cluster_k[f"{data_name}"]

max_iters=100 #Maximum iterations
random_sd = 7
random.seed(random_sd)
sd_list = [random.randint(0,100) for i in range(20)]


print(f'=========={data_name}/{algorithm}/{constrain_type}/{random_sd}==========')
print('==Random seed:',sd_list)
print(f'==cannot_link_path: {cannot_link_path}')
print(f'==must_link_path: {must_link_path}')
print(f'==data_path: {data_path}')
print(f'==file_path: {file_path}')
print(f'==cluster_result_output_path: {cluster_result_output_path}')
print(f'==Number of clusters: {K}')
print(f'==Weight: {P}')



# Calculate Euclidean distance between two points
def euclidean_distance(A,B):
    A = np.asarray(A)
    B = np.asarray(B)
    return np.sqrt(np.sum((A - B)**2, axis=-1))

# Check normalization
def is_normalized(vec, tolerance=1e-6):
    arr = np.asarray(vec)
    return np.isclose(np.linalg.norm(arr), 1.0, atol=tolerance)

# Load data
def load_json(path):
    data = []
    with open(path, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data


# Based on the distance between target vector and word vectors in data dictionary, return the key with minimum distance
def min_key_by_distance(data_dict, target_vector):
    min_key = None
    min_distance = float('inf')
    for key, vector in data_dict.items():
        dist = euclidean_distance(vector, target_vector)
        if dist < min_distance:
            min_key = key
            min_distance = dist
    return min_key

# Get cost matrix
def get_cost_matrix(cl,Centers_mapping,k,cluster_allocation,distance_cc):
    cost_matrix = np.zeros((len(cl), k))
    for i, fixed in enumerate(cl):
        if cluster_allocation[fixed] == 'NOT':
            for j, other in Centers_mapping.items():
                cost_matrix[i, j] = distance_cc[(fixed,j)]
        else:
            for z, o in Centers_mapping.items():
                cost_matrix[i, z] = 9999
            cost_matrix[i, cluster_allocation[fixed]] = 0

    return cost_matrix

# Use KM algorithm to find minimum weight matching
def KM(cost_matrix):
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return col_ind

# Assign clusters to each data point
def assign_clusters(embedding_dict,centers_mapping,ml_constraint_marker,cl_constraint_marker,weight):

    Cluster_allocation = {}
    for i in range(len(embedding_dict)):
        Cluster_allocation[i] = 'NOT'

    mark_points = set()
    for cls_values in cl_constraint_marker.values():
        for cl in cls_values:
            mark_points.update(cl)

    # Calculate distance between all constrained points and centers, dictionary key is (constrained point, center): distance
    # Assign nearest center to all constrained points
    distance_cc = {}
    nearest_center = {}
    for c in mark_points:
        min_key = None
        min_distance = float('inf')
        for center_label,center_embedding in centers_mapping.items():
            d = euclidean_distance(embedding_dict[str(c)], center_embedding)
            distance_cc[(c,center_label)] = d
            if d < min_distance:
                min_key = center_label
                min_distance = d
        nearest_center[c] = min_key


    for i in range(len(embedding_dict)):
        if Cluster_allocation[i] == 'NOT':
            # Check constraints
            # Check if i has ml constraint information
            if len(ml_constraint_marker[i]) != 0:
                print('Error!! Here should only have cl constraints')
            else:
                # Check if i has cl constraint information:
                if len(cl_constraint_marker[i]) != 0:
                    for cl in cl_constraint_marker[i]:
                        cl_copy = copy.deepcopy(cl)
                        while len(cl_copy) > 0:
                            if len(cl_copy) == 1:
                                Cluster_allocation[cl_copy[0]] = nearest_center[cl_copy[0]]
                                break
                            cost_matrix = get_cost_matrix(cl_copy,centers_mapping,K,Cluster_allocation,distance_cc)
                            match_r = KM(cost_matrix)
                            sum_cl_copy_match = 0
                            for m in range(len(cl_copy)):
                                sum_cl_copy_match += distance_cc[(cl_copy[m],match_r[m])]
                            gy_list = []
                            num_y_list = []
                            for c in cl_copy:
                                cl_copy_remove_c = [i for i in cl_copy if i != c]
                                cost_matrix_remove_c = get_cost_matrix(cl_copy_remove_c,centers_mapping,K,Cluster_allocation,distance_cc)
                                match_r_remove_c = KM(cost_matrix_remove_c)
                                num_y = 1
                                for m in range(len(cl_copy_remove_c)):
                                    if match_r_remove_c[m] != match_r[cl_copy.index(cl_copy_remove_c[m])]:
                                        num_y += 1
                                num_y_list.append(num_y)
                                sum_cl_copy_remove_c_match = 0
                                for m in range(len(cl_copy_remove_c)):
                                    sum_cl_copy_remove_c_match += distance_cc[(cl_copy_remove_c[m],match_r_remove_c[m])]
                                gy = sum_cl_copy_match - sum_cl_copy_remove_c_match - distance_cc[(c,nearest_center[c])]
                                gy_list.append(gy)
                            max_gy = max(gy_list)
                            if max_gy < num_y_list[gy_list.index(max_gy)] * weight:
                                for m in range(len(cl_copy)):
                                    Cluster_allocation[cl_copy[m]] = match_r[m]
                                break
                            Cluster_allocation[cl_copy[gy_list.index(max_gy)]] = nearest_center[cl_copy[gy_list.index(max_gy)]]
                            cl_copy.remove(cl_copy[gy_list.index(max_gy)])
                else:
                    min_sort = min_key_by_distance(centers_mapping, embedding_dict[str(i)])
                    Cluster_allocation[i] = min_sort  ##No constraint information, assign to nearest center
    return Cluster_allocation
# Update cluster centers
def update_centers(cluster_allocation,embedding_dict):
    clusters = {}
    for key, cluster_label in cluster_allocation.items():
        if cluster_label not in clusters:
            clusters[cluster_label] = []
        clusters[cluster_label].append(embedding_dict[str(key)])

    cluster_centers = {}
    for cluster_label, points in clusters.items():
        points_array = np.array(points)
        cluster_center = points_array.mean(axis=0)
        l2_norm = np.linalg.norm(cluster_center)
        cluster_center_normalized_l2 = cluster_center / l2_norm
        cluster_centers[cluster_label] = np.array(cluster_center_normalized_l2.tolist())
    return cluster_centers

def compare_dict(dict1, dict2):
    # Check if two dictionaries have same keys
    if set(dict1.keys()) != set(dict2.keys()):
        return False
    for key in dict1.keys():
        if not np.array_equal(dict1[key], dict2[key]):
            return False
    return True

def clustering_accuracy(y_true, y_pred):
    """
    Calculate clustering accuracy (ACC)
    Parameters:
        y_true: True labels (n_samples,)
        y_pred: Cluster labels output by clustering algorithm (n_samples,)
    Returns:
        acc: Clustering accuracy
    """
    labels_true = np.unique(y_true)
    labels_pred = np.unique(y_pred)

    n_class = max(len(labels_true), len(labels_pred))
    confusion_matrix = np.zeros((n_class, n_class), dtype=np.int32)
    for i in range(len(y_true)):
        confusion_matrix[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(-confusion_matrix)
    acc = confusion_matrix[row_ind, col_ind].sum() / len(y_true)
    return acc

def clustering_score(y_true,y_pred):
    return {
        'ACC':clustering_accuracy(y_true, y_pred)*100,
        'NMI':normalized_mutual_info_score(y_true, y_pred)*100,
        'ARI':adjusted_rand_score(y_true,y_pred)*100
    }

def local_search(cls_,mls_,data,embedding_dict,y_true,weight):
    cl_Constraint_marker = {}
    ml_Constraint_marker = {}
    for i in range(len(data)):
        cl_Constraint_marker[i] = []
        ml_Constraint_marker[i] = []
        for j in cls_:
            if i in j:
                cl_Constraint_marker[i].append(j)
        for j in mls_:
            if i in j:
                ml_Constraint_marker[i].append(j)

    all_measures_ls={'ACC':[],'NMI':[],'ARI':[]}
    data_points = list(embedding_dict.values())
    X = np.asarray(data_points)
    process_result = {'sd':[],'acc':[],'nmi':[],'ari':[],'runtime':[],"all":None}
    for sd in sd_list:
        print(f'==={sd}===')
        start_sd = time.perf_counter()
        centers = init_cluster_centers(X,n_clusters=K,y=None, seed_set=None, duplicate_eps=1e-8, random_seed=sd)
        Centers_mapping = {}
        for i in range(len(centers)):
            Centers_mapping[i] = centers[i]

        for iteration in range(max_iters):
            last_centers = Centers_mapping
            Cluster_allocation = assign_clusters(embedding_dict,Centers_mapping,ml_Constraint_marker,cl_Constraint_marker,weight)
            Centers_mapping = update_centers(Cluster_allocation,embedding_dict)
            if compare_dict(Centers_mapping,last_centers):
                break
        ls_y_pred = []
        for i,j in Cluster_allocation.items():
            ls_y_pred.append(j)

        measures_ls = clustering_score(y_true,ls_y_pred)

        for i,j in measures_ls.items():
            all_measures_ls[i].append(j)

        end_sd = time.perf_counter()

        ACC_ls = measures_ls['ACC']
        NMI_ls = measures_ls['NMI']
        ARI_ls = measures_ls['ARI']

        print(f'local search: ACC:{ACC_ls:.2f}% ;NMI:{NMI_ls:.2f} ;ARI:{ARI_ls:.2f}')
        print(f"Time taken: {end_sd - start_sd:.6f} seconds")

        process_result['sd'].append(sd)
        process_result['acc'].append(round(ACC_ls, 2))
        process_result['nmi'].append(round(NMI_ls, 2))
        process_result['ari'].append(round(ARI_ls, 2))
        process_result['runtime'].append(round(end_sd - start_sd, 6))

    ACC_mean = np.mean(all_measures_ls['ACC'])
    ACC_std = np.std(all_measures_ls['ACC'])
    NMI_mean = np.mean(all_measures_ls['NMI'])
    NMI_std = np.std(all_measures_ls['NMI'])
    ARI_mean = np.mean(all_measures_ls['ARI'])
    ARI_std = np.std(all_measures_ls['ARI'])

    print('Average:')
    print(f'ls: ACC_mean:{ACC_mean:.2f} ;ACC_std:{ACC_std:.2f} ;NMI:{NMI_mean:.2f};NMI_std:{NMI_std:.2f};ARI:{ARI_mean:.2f};ARI_std:{ARI_std:.2f}')
    process_result["all"] = f"{ACC_mean:.2f}/{ACC_std:.2f}/{NMI_mean:.2f}/{NMI_std:.2f}/{ARI_mean:.2f}/{ARI_std:.2f}"

    return process_result

# Main
# Load data
data = load_json(data_path)
# Read true labels
y_true_label = []
for i in data:
    y_true_label.append(i['label'])

label_encoder = LabelEncoder()
y_true = label_encoder.fit_transform(y_true_label)

#embeddings
with h5py.File(file_path, 'r') as file:
    if 'embeds' in file:
        dataset = file['embeds'][:]

embedding_dict = {}
for i in range(len(dataset)):
    embedding_dict[str(i)] = np.array(dataset[i])

check_is_normalized = is_normalized(vec=embedding_dict['0'])
print(f'==Check if normalized: {check_is_normalized}')

# Read cls constraints
with open(cannot_link_path, 'r') as json_file:
    cannot_link = json.load(json_file)
print("canot_link:",cannot_link)

# Split cl constraints
# Total number of points
total_points = len(data)
print('==total_points:',total_points)

selected_points = {}
for percentage in percentages:
    selected_count = int((percentage / 100) * total_points)
    selected_points[percentage] = selected_count

result = {}
for percentage, count in selected_points.items():
    selected_so_far = set()
    selected_set = []
    enough_tip = False
    for sublist in cannot_link:
        temp_list = []
        for s in sublist:
            selected_so_far.add(s)
            if len(selected_so_far) <= count:
                temp_list.append(s)
            else:
                enough_tip = True
                break
        selected_set.append(temp_list)
        if enough_tip:
            break
    result[percentage] = selected_set

##Read ml
must_link = [[]]
print("must_link:",must_link)


import pandas as pd
cluster_result = pd.DataFrame(columns=['percent','seed','acc','nmi','ari','runtime','Algorithm'])
cluster_result_all = pd.DataFrame(columns=[2,4,6,8,10,20,30,40])

for pro, cls in result.items():
    start = time.perf_counter()

    cl_proportions = len(set.union(*[set(lst) for lst in cls])) / total_points
    print(f'=====cl proportion: {cl_proportions:.4f}=====')
    print('cl: ', cls)

    result_dict = local_search(cls,[[]],data, embedding_dict, y_true,P)

    end = time.perf_counter()
    print(f"Time taken: {end - start:.6f} seconds")
    for i in range(len(sd_list)):
        new_row = pd.DataFrame({'percent': [pro],
                                'seed': [result_dict['sd'][i]],
                                'acc': [result_dict['acc'][i]],
                                'nmi': [result_dict['nmi'][i]],
                                'ari': [result_dict['ari'][i]],
                                'runtime': [result_dict['runtime'][i]],
                                'Algorithm': [algorithm]}
                               )
        cluster_result = pd.concat([cluster_result, new_row], ignore_index=True)
    cluster_result_all[pro] = [result_dict['all']]

# Save result file
cluster_result.to_csv(cluster_result_output_path, index=False)
cluster_result_all_output_path = cluster_result_output_path[:-4] + "_all" + ".csv"
cluster_result_all.to_csv(cluster_result_all_output_path)
