import time
from collections import defaultdict
import numpy as np
import random
import json
import h5py
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import kmeans_plusplus
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from clustering.utils.kmean_plus_plus import init_cluster_centers


# Parameters
data_name = 'banking77'
data_type = 'small'           #small,small_v3,small_gt,large
constrain_type = 'ml'           #cl,ml,cl_ml
clustering_metric = "cluster_Euclidean"
percentages = [2,4,6,8,10,20,30,40]# Define percentages to select
algorithm = "ls"
P = 0.01

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

print(f'=========={data_name}/{algorithm}/{clustering_metric}/{constrain_type}/{random_sd}==========')
print('==Random seeds:',sd_list)
print(f'==cannot_link_path: {cannot_link_path}')
print(f'==must_link_path: {must_link_path}')
print(f'==data_path: {data_path}')
print(f'==file_path: {file_path}')
print(f'==cluster_result_output_path: {cluster_result_output_path}')
print(f'==Number of clusters: {K}')
print(f'==Weight: {P}')


#Function
def clustering_accuracy(y_true, y_pred):
    """
    Calculate clustering accuracy (Clustering Accuracy, ACC)
    Parameters:
        y_true: True labels (n_samples,)
        y_pred: Cluster labels from clustering algorithm (n_samples,)
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

#Load data
def load_json(path):
    data = []
    with open(path, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

#Get mean embedding
def get_mean_embedding(embedding_dict,point_list):
    embeddings = [embedding_dict[str(key)] for key in point_list]
    average_embedding = np.mean(embeddings, axis=0)
    return average_embedding


#Calculate Euclidean distance between two points
def euclidean_distance(A,B):
    A = np.asarray(A)
    B = np.asarray(B)
    return np.sqrt(np.sum((A - B)**2, axis=-1))


#Return key with minimum distance between target vector and word vectors in data dictionary
def min_key_by_distance(data_dict, target_vector):
    min_key = None
    min_distance = float('inf')
    for key, vector in data_dict.items():
        dist = euclidean_distance(vector, target_vector)
        if dist < min_distance:
            min_key = key
            min_distance = dist
    return min_key


def gj(ml,xj,c_xj,embedding_dict,sum_d,center_mapping,distance_mc):
    ml_remove_xj = [m for m in ml]
    ml_remove_xj.remove(xj)
    mean_ml_remove_xj = get_mean_embedding(embedding_dict,ml_remove_xj)
    c_mean_ml_remove_xj = min_key_by_distance(center_mapping,mean_ml_remove_xj)
    sum_ml_remove_xj = 0
    for m in ml_remove_xj:
        # sum_ml_remove_xj += euclidean_distance(embedding_dict[str(m)],center_mapping[c_mean_ml_remove_xj])
        sum_ml_remove_xj += distance_mc[(m,c_mean_ml_remove_xj)]
    gj_value = sum_d -(sum_ml_remove_xj + euclidean_distance(embedding_dict[str(xj)],center_mapping[c_xj]))
    return gj_value


# Assign clusters to each data point
def assign_clusters(embedding_dict,centers_mapping,ml_constraint_marker,weight):
    Cluster_allocation = {}
    for i in range(len(embedding_dict)):
        Cluster_allocation[i] = 'NOT'
    mark_points = set()
    for mls in ml_constraint_marker.values():
        for ml in mls:
            mark_points.update(ml)

    #Calculate distance between all constrained points and centers, dictionary key is (constrained point, center): distance
    #Assign nearest center to all constrained points
    distance_mc = {}
    nearest_center = {}
    for m in mark_points:
        min_key = None
        min_distance = float('inf')
        for center_label,center_embedding in centers_mapping.items():
            d = euclidean_distance(embedding_dict[str(m)], center_embedding)
            distance_mc[(m,center_label)] = d
            if d < min_distance:
                min_key = center_label
                min_distance = d
        nearest_center[m] = min_key


    for i in range(len(Cluster_allocation)):
        if Cluster_allocation[i] == 'NOT':

            # Check if i has ml constraint information
            if len(ml_constraint_marker[i]) != 0:

                mls_copy = [m for m in ml_constraint_marker[i][0]]
                check_point = False
                while len(mls_copy) > 0:

                    if len(mls_copy) == 1:
                        Cluster_allocation[mls_copy[0]] = nearest_center[mls_copy[0]]
                        break

                    mean_ml_copy = get_mean_embedding(embedding_dict,mls_copy)
                    c_mean_ml_copy = min_key_by_distance(centers_mapping, mean_ml_copy)
                    sum_d = 0
                    for m in mls_copy:
                        # sum_d += euclidean_distance(embedding_dict[str(m)],centers_mapping[c_mean_ml_copy])
                        sum_d += distance_mc[(m,c_mean_ml_copy)]

                    g_list = []
                    for xj in mls_copy:
                        g = gj(mls_copy,xj,nearest_center[xj],embedding_dict,sum_d,centers_mapping,distance_mc)
                        g_list.append(g)
                    # print(f'g:{g_list}')
                    max_gj = max(g_list)

                    if max_gj < weight:
                        for m in mls_copy:
                            Cluster_allocation[m] = c_mean_ml_copy
                        check_point = True

                    if check_point:
                        break
                    else:
                        Cluster_allocation[mls_copy[g_list.index(max_gj)]] = nearest_center[mls_copy[g_list.index(max_gj)]]
                        mls_copy.remove(mls_copy[g_list.index(max_gj)])

            else:
                min_key = min_key_by_distance(centers_mapping, embedding_dict[str(i)])
                Cluster_allocation[i] = min_key

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

def local_search(Mls_,y_true,embedding_dict,max_iters,weight):
    embedding_dict_copy = {}
    for i,j in embedding_dict.items():
        embedding_dict_copy[i] = j

    embeddings_use_init = embedding_dict.copy()#For initialization

    clear_mls = {}
    fusion_points = list(Mls_.keys())
    intermediate_fusion_point = []

    for point, ml in Mls_.items():
        ml_copy = [i for i in ml]
        m_remove = []
        for m in ml_copy:
            if m in fusion_points:
                for l in Mls_[m]:
                    ml_copy.append(l)
                m_remove.append(m)
                intermediate_fusion_point.append(m)

        for i in m_remove:
            ml_copy.remove(i)

        if len(ml_copy) >= 2:
            clear_mls[point] = ml_copy

    for point in intermediate_fusion_point:
        try:
            del clear_mls[point]
        except KeyError:
            continue


    Fusion_point_weight = {}

    for i, j in clear_mls.items():
        avg_e = get_mean_embedding(embedding_dict, j)
        for m in j:
            embeddings_use_init[m] = avg_e
    print('embeddings_use_init',len(embeddings_use_init))


    for i, j in clear_mls.items():
        avg_e = get_mean_embedding(embedding_dict,j)
        embedding_dict_copy[i] = avg_e

    mark_point = set()
    for point, ml in clear_mls.items():
        mark_point.update(ml)

    for point in mark_point:
        del embedding_dict_copy[str(point)]

    point_weight = {}
    for point, e in embedding_dict_copy.items():
        if int(point) not in Fusion_point_weight.keys():
            point_weight[int(point)] = 1
        else:
            point_weight[int(point)] = Fusion_point_weight[int(point)]

    data_point_weight = np.array(list(point_weight.values()))

    all_measures_ls = {'ACC': [], 'NMI': [], 'ARI': []}

    ml_Constraint_marker = {}
    for point in range(len(data)):
        ml_Constraint_marker[point] = []

    for id,ml in clear_mls.items():
        ml = set([int(i) for i in ml])
        for m in ml:
            ml_Constraint_marker[m].append(ml)


    data_points = list(embedding_dict.values())
    X = np.array(data_points)

    X_init = np.array(list(embeddings_use_init.values()))
    process_result = {'sd':[],'acc':[],'nmi':[],'ari':[],'runtime':[],"all":None}
    for sd in sd_list:
        start_sd = time.perf_counter()
        print(f"==={sd}===")
        # centers, indices = kmeans_plusplus(X, n_clusters=K, random_state=sd)
        centers = init_cluster_centers(X_init, n_clusters=K, y=None, seed_set=None, duplicate_eps=1e-8, random_seed=sd)
        # sample_weight=data_point_weight
        Centers_mapping = {}
        for i in range(len(centers)):
            Centers_mapping[i] = np.asarray(centers[i])

        for iteration in range(max_iters):
            last_centers = Centers_mapping
            Cluster_allocation = assign_clusters(embedding_dict,Centers_mapping,ml_Constraint_marker,weight)
            Centers_mapping = update_centers(Cluster_allocation,embedding_dict)
            if compare_dict(Centers_mapping,last_centers):
                break
        ls_p_y_pred = []
        for i,j in Cluster_allocation.items():
            ls_p_y_pred.append(j)
        measures_ls_p = clustering_score(y_true, ls_p_y_pred)

        for i, j in measures_ls_p.items():
            all_measures_ls[i].append(j)

        end_sd = time.perf_counter()

        ACC = measures_ls_p['ACC']
        NMI = measures_ls_p['NMI']
        ARI = measures_ls_p['ARI']

        print(f'==ls: ACC:{ACC:.2f}% ;NMI:{NMI:.2f} ;ARI:{ARI:.2f}')
        print(f"Time taken: {end_sd - start_sd:.6f} seconds")

        process_result['sd'].append(sd)
        process_result['acc'].append(round(ACC, 2))
        process_result['nmi'].append(round(NMI, 2))
        process_result['ari'].append(round(ARI, 2))
        process_result['runtime'].append(round(end_sd - start_sd, 6))


    ACC_mean = np.mean(all_measures_ls['ACC'])
    ACC_std = np.std(all_measures_ls['ACC'])
    NMI_mean = np.mean(all_measures_ls['NMI'])
    NMI_std = np.std(all_measures_ls['NMI'])
    ARI_mean = np.mean(all_measures_ls['ARI'])
    ARI_std = np.std(all_measures_ls['ARI'])

    print('Average case:')
    print(
        f'Constrained clustering: ACC_mean:{ACC_mean:.2f} ;ACC_std:{ACC_std:.2f} ;NMI:{NMI_mean:.2f};NMI_std:{NMI_std:.2f};ARI:{ARI_mean:.2f};ARI_std:{ARI_std:.2f}')
    process_result["all"] = f"{ACC_mean:.2f}/{ACC_std:.2f}/{NMI_mean:.2f}/{NMI_std:.2f}/{ARI_mean:.2f}/{ARI_std:.2f}"

    return process_result


#main
data = load_json(data_path)

#Read true labels
y_true_label = []
for i in data:
    y_true_label.append(i['label'])
label_encoder = LabelEncoder()
y_true = label_encoder.fit_transform(y_true_label)

# Read embeddings
with h5py.File(file_path, 'r') as file:
    if 'embeds' in file:
        dataset = file['embeds'][:]

embedding_dict = {}
for i in range(len(dataset)):
    embedding_dict[str(i)] = np.array(dataset[i])

##Read ml markers
must_link = []
with open(must_link_path, 'r') as json_file:
    Mls_Data = json.load(json_file)

Mls = {}
for grid,_ in Mls_Data.items():
    for i,j in _.items():
        Mls[i] = j
print(Mls)

#Split Mls
#Total number of points
total_points = len(data)
print('total_points:',total_points)

# Calculate number of points to select for each percentage
selected_points = {}
for percentage in percentages:
    selected_count = int((percentage / 100) * total_points)
    selected_points[percentage] = selected_count


# Save selected point sets for each percentage
result = {}

# Traverse from first list, select points in order
for percentage, count in selected_points.items():

    selected_so_far = set()
    selected_dict = {}
    enough_tip = False
    for num,sublist in Mls.items():
        temp_list = []
        for s in sublist:
            selected_so_far.add(s)
            if len(selected_so_far) <= count:
                temp_list.append(s)
            else:
                enough_tip = True
                break

        selected_dict[num] = temp_list
        if enough_tip:
            break
    result[percentage] = selected_dict

import pandas as pd
cluster_result = pd.DataFrame(columns=['percent','seed','acc','nmi','ari','runtime','Algorithm'])
cluster_result_all = pd.DataFrame(columns=[2,4,6,8,10,20,30,40])

for pro,mls in result.items():
    start = time.perf_counter()
    mls_list = list(mls.values())
    ml_proportions = len(set.union(*[set(lst) for lst in mls_list])) / total_points
    print(f'======ml proportion: {ml_proportions:.4f}=====')
    mls = {key: value for key, value in mls.items() if len(value) >= 2}
    result_dict = local_search(mls,y_true,embedding_dict,100,P)
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

cluster_result.to_csv(cluster_result_output_path, index=False)
cluster_result_all_output_path = cluster_result_output_path[:-4] + "_all" + ".csv"
cluster_result_all.to_csv(cluster_result_all_output_path)







