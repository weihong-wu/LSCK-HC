import copy
import time
from collections import defaultdict
import numpy as np
import random
import json
import h5py
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from clustering.utils.kmean_plus_plus import init_cluster_centers

# Parameters
data_name = 'banking77'
data_type = 'small'                  # small,small_v3,small_gt,large
constrain_type = 'cl_ml'             # cl,ml,cl_ml
clustering_metric = "cluster_Euclidean"
percentages = [2,4,6,8,10,20,30,40]  # Define percentages to select
algorithm = "LSCK-HC"
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

# Read confidence scores
data_confidence_path = 'code/clustering/utils/ml_confidence.json'
with open(data_confidence_path, 'r') as json_file:
    data_confidence = json.load(json_file)

confs  = data_confidence[f"{data_name}_{data_type}"]
grid_confidence = {}
for g,conf in confs.items():
    grid_confidence[int(g)] = conf

print(f'=========={data_name}/{algorithm}/{clustering_metric}/{constrain_type}/{random_sd}==========')
print('==Random seeds:',sd_list)
print(f'==cannot_link_path: {cannot_link_path}')
print(f'==must_link_path: {must_link_path}')
print(f'==data_path: {data_path}')
print(f'==file_path: {file_path}')
print(f'==cluster_result_output_path: {cluster_result_output_path}')
print(f'==Number of clusters: {K}')
print(f'==Weight: {P}')
print(f"==grid_confidence:{grid_confidence}")


#Function
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

#Load data
def load_json(path):
    data = []
    with open(path, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

#Get mean embedding
def get_mean_embedding(embeddings,point_list):
    embs = [embeddings[str(key)] for key in point_list]
    average_embedding = np.mean(embs, axis=0)
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


def merge_or_not(embeddings,center_mapping,Gij,Gil,weight):
    Gij_mean_embedding = get_mean_embedding(embeddings,Gij)
    Gil_mean_embedding = get_mean_embedding(embeddings,Gil)

    Gij_Gil = Gij + Gil
    Gij_Gil_mean_embedding = get_mean_embedding(embeddings,Gij_Gil)
    Gij_Gil = list(map(str, Gij_Gil))
    Gij_Gil_embedding = []
    for a in Gij_Gil:
        Gij_Gil_embedding.append(embeddings[str(a)])
    c = min_key_by_distance(center_mapping,Gij_Gil_mean_embedding)
    cj = min_key_by_distance(center_mapping,Gij_mean_embedding)
    cl = min_key_by_distance(center_mapping,Gil_mean_embedding)

    grade1 = (weight + euclidean_distance(Gil_mean_embedding,center_mapping[cl])) * int(len(Gil)) + int(len(Gij)) * (weight + euclidean_distance(Gij_mean_embedding,center_mapping[cj]))
    grade2 = np.sum(euclidean_distance(Gij_Gil_embedding,center_mapping[c]))

    if grade1 > grade2:
        return True
    else:
        return False


# Assign clusters to each data point
def ml_cutting(embeddings,centers_mapping,ml_constraint_marker,weight):

    embedding_dict_p = embeddings.copy()

    cutted_points = set()

    for i in ml_constraint_marker.keys():
        if len(ml_constraint_marker[i]) != 0:
            if i in cutted_points:
                continue
            cutted_points.update(ml_constraint_marker[i][0])

            point_nearest_centers = {}
            for m in ml_constraint_marker[i][0]:
                point_nearest_centers[m] = min_key_by_distance(centers_mapping,embeddings[str(m)])

            G_dict = defaultdict(list)
            for key, value in point_nearest_centers.items():
                G_dict[value].append(key)
            G = sorted(G_dict.values(), key=lambda x: len(x),reverse=True)
            G_copy =  G.copy()
            merges = [False] * len(G)
            for g in range(len(G)):
                if not merges[g]:
                    merges[g] = True
                    G_copy.remove(G[g])
                    G_copy = sorted(G_copy, key=len, reverse=True)
                    merged_sets = []
                    for set_ in G_copy:
                        if merge_or_not(embeddings,centers_mapping,G[g],set_,weight):
                            merged_sets.append(set_)
                            if set_ in G:
                                merges[G.index(set_)] = True
                            G[g] = G[g] + set_

                    G_copy.append(G[g])
                    if len(merged_sets) != 0:
                        for set_ in merged_sets:
                            G_copy.remove(set_)

            for set_ in G_copy:
                set_mean_embedding = get_mean_embedding(embeddings,set_)
                for s in set_:
                    embedding_dict_p[str(s)] = set_mean_embedding
    return embedding_dict_p
#Get cost matrix
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

#Use KM algorithm to find minimum weight matching
def KM(cost_matrix):
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return col_ind

# Assign clusters to each data point
def assign_clusters(embeddings,centers_mapping,cl_constraint_marker,weight):

    Cluster_allocation = {}
    for i in embeddings.keys():
        Cluster_allocation[int(i)] = 'NOT'

    mark_points = set()
    for cls in cl_constraint_marker.values():
        for cl in cls:
            mark_points.update(cl)

    #Calculate distance between all constrained points and centers, dictionary key is (constrained point, center): distance
    #Assign nearest center to all constrained points
    distance_cc = {}
    nearest_center = {}
    for c in mark_points:
        min_key = None
        min_distance = float('inf')
        for center_label,center_embedding in centers_mapping.items():
            d = euclidean_distance(embeddings[str(c)], center_embedding)
            distance_cc[(c,center_label)] = d
            if d < min_distance:
                min_key = center_label
                min_distance = d
        nearest_center[c] = min_key


    for i in Cluster_allocation.keys():
        if Cluster_allocation[i] == 'NOT':
            # Check if i has cl constraint information:
            if len(cl_constraint_marker[i]) != 0:
                for cl in cl_constraint_marker[i]:
                    cl_copy = copy.deepcopy(cl)
                    while len(cl_copy) > 0:
                        if len(cl_copy) == 1:
                            Cluster_allocation[cl_copy[0]] = nearest_center[cl_copy[0]]
                            break
                        cost_matrix = get_cost_matrix(cl_copy,centers_mapping,K,Cluster_allocation,distance_cc)
                        # print(cost_matrix)
                        match_r = KM(cost_matrix)
                        sum_cl_copy_match = 0
                        for m in range(len(cl_copy)):
                            # sum_cl_copy_match += euclidean_distance(embeddings[str(cl_copy[m])],centers_mapping[match_r[m]])
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
                min_sort = min_key_by_distance(centers_mapping, embeddings[str(i)])
                Cluster_allocation[i] = min_sort  # No constraint information, assign to nearest center
    return Cluster_allocation

# Update cluster centers
def update_centers(cluster_allocation,embedding_dict_p_,pw):

    # Create a dictionary to store all points for each cluster

    clusters = {}
    clusters_point_weight = {}

    for key, cluster_label in cluster_allocation.items():
        if cluster_label not in clusters:
            clusters[cluster_label] = []
        if cluster_label not in clusters_point_weight:
            clusters_point_weight[cluster_label] = []
        clusters[cluster_label].append(embedding_dict_p_[str(key)])
        clusters_point_weight[cluster_label].append(pw[key])
    cluster_centers = {}
    for cluster_label, points in clusters.items():
        points_array = np.array(points)
        cluster_center = np.average(points_array, axis=0, weights=clusters_point_weight[cluster_label])
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


def lsc_kmmp(Cls_,Mls_,ml_confidence_,y_true,embeddings,max_iters,weight):

    embedding_dict_copy = embeddings.copy()
    embeddings_use_init = embeddings.copy()# For initialization

    ml_confidence_keys = ml_confidence_.keys()

    #Clean intermediate points from merging
    clear_mls = {}
    fusion_points = list(Mls_.keys())
    intermediate_fusion_point = []

    for point, ml in Mls_.items():
        confidences_list = []
        try:
            point_confidence = ml_confidence_[tuple([int(m) for m in ml])]
        except:
            for _ in ml_confidence_keys:
                if set(_) & set(ml):
                    point_confidence = ml_confidence_[tuple([int(m) for m in _])]
        confidences_list.append(point_confidence)
        ml_copy = [i for i in ml]
        m_remove = []
        for m in ml_copy:
            if m in fusion_points:
                confidences_list.append(ml_confidence_[tuple([int(m) for m in Mls_[m]])])

                for l in Mls_[m]:
                    ml_copy.append(l)
                m_remove.append(m)
                intermediate_fusion_point.append(m)

        for i in m_remove:
            ml_copy.remove(i)

        if len(ml_copy) >= 2:
            clear_mls[point] = ml_copy
            if False in confidences_list:
                ml_confidence_[tuple([int(m) for m in ml_copy])] = False
            else:
                ml_confidence_[tuple([int(m) for m in ml_copy])] = True

    for point in intermediate_fusion_point:
        try:
            del clear_mls[point]
        except KeyError:
            continue


    high_confidence_ml = {}
    low_confidence_ml = {}
    for num, ml in clear_mls.items():
        ml_figure = [int(m) for m in ml]
        if ml_confidence_[tuple(ml_figure)]:
            high_confidence_ml[num] = ml
        else:
            low_confidence_ml[num] = ml

    clear_high_confidence_ml = {}
    clear_low_confidence_ml = {}
    for num, ml in high_confidence_ml.items():
        temp = []
        for m in ml:
            if int(m) <= len(data):
                temp.append(m)
        clear_high_confidence_ml[num] = temp

    for num, ml in low_confidence_ml.items():
        temp = []
        for m in ml:
            if int(m) <= len(data):
                temp.append(m)
        clear_low_confidence_ml[num] = temp


    print(high_confidence_ml)
    print(low_confidence_ml)

    # For weighted initialization, replace points in constraints with ml
    for i, j in clear_high_confidence_ml.items():
        avg_e = get_mean_embedding(embeddings, j)
        for m in j:
            embeddings_use_init[m] = avg_e
    print('embeddings_use_init',len(embeddings_use_init))

    Fusion_point_weight = {}  # Weights for merged points
    mapping_dict = {}  # Mapping for marked points
    for point,ml in clear_high_confidence_ml.items():
        Fusion_point_weight[int(point)] = len(ml)
        for m in ml:
            mapping_dict[int(m)] = int(point)

    for i, j in clear_high_confidence_ml.items():
        avg_e = get_mean_embedding(embeddings,j)
        embedding_dict_copy[i] = avg_e

    #Marked points
    mark_point = set()
    for point, ml in clear_high_confidence_ml.items():
        mark_point.update(ml)

    # Remove marked points
    for point in mark_point:
        del embedding_dict_copy[str(point)]

    # Calculate weights
    point_weight = {}
    for point, e in embedding_dict_copy.items():
        if int(point) not in Fusion_point_weight.keys():
            point_weight[int(point)] = 1
        else:
            point_weight[int(point)] = Fusion_point_weight[int(point)]

    all_measures = {'ACC': [], 'NMI': [], 'ARI': []}

    # Constraint markers:
    ml_Constraint_marker = {}
    for point in embedding_dict_copy.keys():
        ml_Constraint_marker[int(point)] = []

    for id,ml in clear_low_confidence_ml.items():
        ml = set([int(i) for i in ml])
        for m in ml:
            ml_Constraint_marker[m].append(ml)

    # Process cl accordingly
    clear_cls = []
    for cl in Cls_:
        temp_cl = []
        for c in cl:
            if str(c) in mark_point:
                temp_cl.append(mapping_dict[c])
            else:
                temp_cl.append(c)
        clear_cls.append(temp_cl)

    # Constraint markers:
    cl_Constraint_marker = {}
    for i in embedding_dict_copy.keys():
        cl_Constraint_marker[int(i)] = []
        for j in clear_cls:
            if int(i) in j:
                cl_Constraint_marker[int(i)].append(j)

    X_init = np.array(list(embeddings_use_init.values()))

    process_result = {'sd':[],'acc':[],'nmi':[],'ari':[],'runtime':[],"all":None}


    for sd in sd_list:
        start_sd = time.perf_counter()
        print(f"==={sd}===")
        centers = init_cluster_centers(X_init, n_clusters=K, y=None, seed_set=None, duplicate_eps=1e-8, random_seed=sd)

        Centers_mapping = {}
        for i in range(len(centers)):
            Centers_mapping[i] = np.asarray(centers[i])

        for iteration in range(max_iters):
            last_centers = Centers_mapping
            embedding_dict_copy_p = ml_cutting(embedding_dict_copy,Centers_mapping,ml_Constraint_marker,weight)
            Cluster_allocation = assign_clusters(embedding_dict_copy_p,Centers_mapping,cl_Constraint_marker,weight)
            Centers_mapping = update_centers(Cluster_allocation,embedding_dict_copy_p,point_weight)
            if compare_dict(Centers_mapping,last_centers):
                break

        lsc_kmmp_y_pred = [None] * len(data)
        for p in range(len(lsc_kmmp_y_pred)):
            if p in Cluster_allocation.keys():
                lsc_kmmp_y_pred[p] = Cluster_allocation[p]
            else:
                lsc_kmmp_y_pred[p] = Cluster_allocation[mapping_dict[p]]
        measures = clustering_score(y_true, lsc_kmmp_y_pred)

        for i, j in measures.items():
            all_measures[i].append(j)
        end_sd = time.perf_counter()

        ACC = measures['ACC']
        NMI = measures['NMI']
        ARI = measures['ARI']

        print(f'ls_kmmp: ACC:{ACC:.2f}% ;NMI:{NMI:.2f} ;ARI:{ARI:.2f}')
        print(f"Time taken: {end_sd - start_sd:.6f} seconds")

        process_result['sd'].append(sd)
        process_result['acc'].append(round(ACC,2))
        process_result['nmi'].append(round(NMI,2))
        process_result['ari'].append(round(ARI,2))
        process_result['runtime'].append(round(end_sd - start_sd,6))


    ACC_mean = np.mean(all_measures['ACC'])
    ACC_std = np.std(all_measures['ACC'])
    NMI_mean = np.mean(all_measures['NMI'])
    NMI_std = np.std(all_measures['NMI'])
    ARI_mean = np.mean(all_measures['ARI'])
    ARI_std = np.std(all_measures['ARI'])

    print('Average case:')
    print(f'Constrained clustering: ACC_mean:{ACC_mean:.2f} ;ACC_std:{ACC_std:.2f} ;NMI:{NMI_mean:.2f};NMI_std:{NMI_std:.2f};ARI:{ARI_mean:.2f};ARI_std:{ARI_std:.2f}')
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
    embedding_dict[str(i)] = np.asarray(dataset[i])

#Total number of points
total_points = len(data)
print('==total_points:',total_points)

#Read cls constraints
with open(cannot_link_path, 'r') as json_file:
    cannot_link = json.load(json_file)
cannot_link = [cl for cl in cannot_link if len(cl) >= 2]
print("==canot_link:",cannot_link)


#Read must_link
with open(must_link_path, 'r') as json_file:
    Mls_Data = json.load(json_file)

mls_confidence = {}  # Constraint confidence
for grid, mls_dict in Mls_Data.items():
    confidence2 = grid_confidence[int(grid)][0]
    confidence_over2 = grid_confidence[int(grid)][1]
    num_pair = 0
    num_set = 0
    for _, ml in mls_dict.items():
        figure_ml = [int(m) for m in ml]
        if len(figure_ml) == 2:
            if (num_pair < int(confidence2)):
                mls_confidence[tuple(figure_ml)] = True
            else:
                mls_confidence[tuple(figure_ml)] = False
            num_pair += 1
        else:
            if (num_set < int(confidence_over2)):
                mls_confidence[tuple(figure_ml)] = True
            else:
                mls_confidence[tuple(figure_ml)] = False
            # mls_confidence[tuple(figure_ml)] = confidence_over2
            num_set += 1

Mls = {}
for grid,mls_dict in Mls_Data.items():
    for num,ml in mls_dict.items():
        Mls[num] = ml

collected_mls = []
constrains_links = []
for cl in cannot_link:
    constrains_link = {}
    for c in cl:
        constrains_link[c] = {}
        for num,ml in Mls.items():
            if num not in collected_mls:
                if str(c) in ml:
                    constrains_link[c][num] = ml
                    collected_mls.append(num)
    constrains_links.append(constrains_link)

sort_dict = {}
for site,constrains_link in enumerate(constrains_links):
    point_sets = set()
    for num,ml in constrains_link.items():
        point_sets.add(str(num))
        for _,m in ml.items():
            point_sets.update(m)
    sort_dict[site] = len(point_sets)

# Sort by value descending, get keys
sorted_keys_desc = [k for k, v in sorted(sort_dict.items(), key=lambda item: item[1], reverse=True)]

sorted_constrains_links = []
for keys in sorted_keys_desc:
    sorted_constrains_links.append(constrains_links[keys])

remain_ml = {"remain_ml":{}}
for num,ml in Mls.items():
    if num not in collected_mls:
        remain_ml["remain_ml"][num] = ml

sorted_constrains_links.append(remain_ml)
print(sorted_constrains_links)

#Split by percentage
# Calculate number of points to select for each percentage
selected_points = {}
for percentage in percentages:
    selected_count = int((percentage / 100) * total_points)
    selected_points[percentage] = selected_count

#Cut ml
# Save selected point sets for each percentage
result = {}

for percentage, count in selected_points.items():
    selected_so_far = set()
    selected_set = []
    enough_tip = False
    for sublist in sorted_constrains_links:
        temp_list = {}
        for c,ml in sublist.items():
            if len(selected_so_far) <= count:
                selected_so_far.add(str(c))
                temp_list[c] = {}
                for _,m in ml.items():
                    temp_list[c][_] = []
                    for p in m:
                        if len(selected_so_far) <= count:
                            selected_so_far.add(str(p))
                            temp_list[c][_].append(p)
                        else:
                            enough_tip = True
                            break

                    if enough_tip:
                        break
            else:
                enough_tip = True
                break
        selected_set.append(temp_list)
        if enough_tip:
            break
    result[percentage] = selected_set

import pandas as pd
cluster_result = pd.DataFrame(columns=['percent','seed','acc','nmi','ari','runtime','Algorithm'])
cluster_result_all = pd.DataFrame(columns=[2,4,6,8,10,20,30,40])

for pro,constrains in result.items():
    start = time.perf_counter()
    cls = []
    mls = {}
    for con in constrains:
        cl = []
        for c, ml in con.items():
            cl.append(c)
            for p, m in ml.items():
                mls[p] = m
        cls.append(cl)
    cls = [cl for cl in cls if len(cl) >= 2]
    print(f"======Constraint proportion: {pro}=====")
    print(f"==cls:{cls}")
    print(f"==mls:{mls}")

    result_dict = lsc_kmmp(cls, mls, mls_confidence, y_true, embedding_dict, 100, P)
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
