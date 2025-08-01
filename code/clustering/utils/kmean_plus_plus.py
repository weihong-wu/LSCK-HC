
import random
import time
import h5py
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.extmath import row_norms


def init_cluster_centers(X,n_clusters,init = 'kmeans++', y=None, seed_set=None, duplicate_eps=1e-8, random_seed=0):
    random_state = np.random.RandomState(random_seed)
    assert n_clusters <= len(X), breakpoint()
    x_squared_norms = row_norms(X, squared=True)

    if init == "random":
        remaining_row_idxs = list(range(len(X)))
        seeds = np.empty((n_clusters, X.shape[1]))
        seeds[:] = np.nan
        for i in range(n_clusters):
            while True:
                sampled_idx = random.choice(remaining_row_idxs)
                sampled_vector = X[sampled_idx]
                distance_to_seeds = np.linalg.norm(seeds - sampled_vector, axis=1)
                unique = False
                if i == 0:
                    unique = True
                else:
                    duplicate_found = np.min(
                        distance_to_seeds[np.logical_not(np.isnan(distance_to_seeds))]) < duplicate_eps
                    if not duplicate_found:
                        unique = True
                remaining_row_idxs.remove(sampled_idx)
                if unique:
                    seeds[i] = sampled_vector
                    break
    else:
        timer_dict = {}
        timer = time.perf_counter()
        kpp_start = timer

        # Use k-means++ (https://en.wikipedia.org/wiki/K-means%2B%2B#Improved_initialization_algorithm) to
        # initialize the cluster centers.

        # Using the same method as described by Arthur and Vassilvitskii (2007), we choose `n_local_trials`
        # top candidates for the next cluster center and select the one among these which will most reduce
        # the sum total distance to the existing set of cluster centers.
        n_local_trials = 2 + int(np.log(n_clusters))

        # This is an expensive >quadratic operation which will be very slow for large datasets.
        cluster_seeds = []
        remaining_row_idxs = list(range(len(X)))
        if seed_set is None:
            # Pick initial cluster center.
            sampled_idx = random_state.choice(remaining_row_idxs)
            seed_set = [X[sampled_idx]]
            cluster_seeds.append(X[sampled_idx])
        else:
            cluster_seeds.extend(seed_set)

        closest_dist_sq_all = euclidean_distances(seed_set, X, Y_norm_squared=x_squared_norms, squared=True)
        timer_dict["Initial Euclidean Distances"] = time.perf_counter() - timer
        timer = time.perf_counter()

        timer_dict["Pairwise Euclidean Distances"] = 0
        timer_dict["Compute candidate potentials"] = 0
        closest_dist_sq = np.min(closest_dist_sq_all, axis=0)
        for i in range(len(cluster_seeds), n_clusters):
            nearest_distances_normalized = closest_dist_sq / sum(closest_dist_sq)
            assert len(nearest_distances_normalized.shape) == 1
            assert len(remaining_row_idxs) == len(nearest_distances_normalized)

            # Try out the top 'n_local_trials' choices for the next seed, and choose the one with least
            # average distance to other points in the dataset.
            candidate_ids = random_state.choice(remaining_row_idxs, p=nearest_distances_normalized, size=n_local_trials)
            start = time.perf_counter()
            distance_to_candidates = euclidean_distances(X[candidate_ids], X, Y_norm_squared=x_squared_norms,
                                                         squared=True)
            timer_dict["Pairwise Euclidean Distances"] += time.perf_counter() - start

            start = time.perf_counter()
            min_remaining_distance_to_candidates = np.minimum(closest_dist_sq, distance_to_candidates)
            candidate_potentials = min_remaining_distance_to_candidates.sum(axis=1)
            best_candidate = np.argmin(candidate_potentials)
            timer_dict["Compute candidate potentials"] += time.perf_counter() - start

            # The `closest_dist_sq` array should contain the distance from each point in
            # the dataset to its closest seed point.
            closest_dist_sq = min_remaining_distance_to_candidates[best_candidate]
            cluster_seeds.append(X[candidate_ids[best_candidate]])
        seeds = np.vstack(cluster_seeds)
        timer_dict["Total K-Means++ time"] = time.perf_counter() - kpp_start
    return seeds


