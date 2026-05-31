from __future__ import annotations
from ..classes.Satellite import *
from collections import deque
import pandas as pd
import numpy as np
import copy
import heapq

def count_unique_edges(sats: list[Satellite], i: int):
    edge_set = set((min(sat.id, neighbor_id), max(sat.id, neighbor_id)) for sat in sats for neighbor_id in sat.list_coordinates[i].neighbor_ids)
    return len(edge_set)

def count_unreachable_satellites(sats: list[Satellite], i: int):
    degrees = np.array([len(sat.list_coordinates[i].neighbors) for sat in sats])
    return np.sum(degrees == 0)

def degree_distribution(sats: list[Satellite], i: int):
    degrees = np.array([len(sat.list_coordinates[i].neighbors) for sat in sats])
    unique_degrees, counts = np.unique(degrees, return_counts=True)
    return dict(zip(unique_degrees, counts))

def average_degree(sats: list[Satellite], i: int):
    degrees = np.array([len(sat.list_coordinates[i].neighbors) for sat in sats])
    return np.mean(degrees)

def max_edges_possible(sats: list[Satellite]):
    n = len(sats)
    return n*(n-1)/2

def density(sats: list[Satellite], i: int, nb_unique_edges: int = None, max_edges: int = None):
    nb_l = nb_unique_edges if nb_unique_edges is not None else count_unique_edges(sats, i)
    max_l = max_edges if max_edges is not None else max_edges_possible(sats)
    return nb_l / max_l

def label_matrix(sats: list[Satellite], data, key: "name" | "id"):
    keys = [getattr(sat, key) for sat in sats]
    return pd.DataFrame(data, index=keys, columns=keys)

def relabel_matrix(df: pd.DataFrame, sats: list[Satellite], from_key: str, to_key: str):
    mapping = {getattr(sat, from_key): getattr(sat, to_key) for sat in sats}
    return df.rename(index=mapping, columns=mapping)

def adjacency_matrix(sats: list[Satellite], i: int, key="id"):
    id_to_idx = {sat.id: idx for idx, sat in enumerate(sats)}
    
    n = len(sats)
    adj_matrix = np.zeros((n, n), dtype=int)

    for idx, sat in enumerate(sats):
        neighbor_indices = [id_to_idx[nid] for nid in sat.list_coordinates[i].neighbor_ids]
        adj_matrix[idx, neighbor_indices] = 1
        adj_matrix[neighbor_indices, idx] = 1
    
    return label_matrix(sats, adj_matrix, key)
    
def degree_matrix(sats: list[Satellite], i: int, key="id"):
    return label_matrix(sats, np.diag([len(sat.list_coordinates[i].neighbors) for sat in sats]), key)

def weight_matrix(sats: list[Satellite], i: int, key="id"):
    id_to_idx = {sat.id: idx for idx, sat in enumerate(sats)}
    n = len(sats)
    
    w_matrix = np.zeros((n, n), dtype=int)
    for idx, sat in enumerate(sats):
        for neighbor in sat.list_coordinates[i].neighbors:
            w_matrix[idx, id_to_idx[neighbor.id]] = 10 # TODO Weight formula between each sat
            
    return label_matrix(sats, w_matrix, key)  
    
def laplacian_matrix(sats: list[Satellite], i: int, degree_mat = None, adjacency_mat = None, key="id"):
    return (degree_mat if degree_mat is not None else degree_matrix(sats, i, key)) - (adjacency_mat if adjacency_mat is not None else adjacency_matrix(sats, i, key))

def fielder_value(sats: list[Satellite], i: int, laplacian_mat = None):
    eigenvalues = np.linalg.eigvalsh((laplacian_mat if laplacian_mat is not None else laplacian_matrix(sats, i)).to_numpy())
    fielder_value = eigenvalues[1]
    return fielder_value if fielder_value > 1e-12 else 0.0

def dijkstra(sats: list[Satellite], i: int, start_sat_id, weight_mat = None):
    sats_dict = {sat.id: sat for sat in sats}
    w_matrix = weight_mat if weight_mat is not None else weight_matrix(sats, i)
    
    distances = {sat.id: [float('inf'), 0] for sat in sats}
    distances[start_sat_id] = [0, 1]

    heap = [(0, start_sat_id)]

    while heap:
        # Pop in heap
        current_dist, candidate = heapq.heappop(heap)

        # If better distance stored, skip. Called multiple neighbors.
        if current_dist > distances[candidate][0]:
            continue

        # Extract stored data
        sat = sats_dict[candidate]
        dist_sat, nb_best_paths_sat = distances[sat.id]

        # For each neighbor, determine the cumulated weight to reach them with the new distance we've been called & set with
        for neighbor_id in sat.list_coordinates[i].neighbor_ids:
            dist_to_reach_neighbor = dist_sat + w_matrix.loc[sat.id, neighbor_id]

            # If it is actually better than what it had
            if dist_to_reach_neighbor < distances[neighbor_id][0]:
                distances[neighbor_id] = [dist_to_reach_neighbor, nb_best_paths_sat]
                heapq.heappush(heap, (dist_to_reach_neighbor, neighbor_id))

            # If it was equal
            elif dist_to_reach_neighbor == distances[neighbor_id][0]:
                distances[neighbor_id][1] += nb_best_paths_sat
                
    return distances

def average_path_length(sats: list[Satellite], i: int, weight_mat = None, all_dijkstras = None):
    w_matrix = weight_mat if weight_mat is not None else weight_matrix(sats, i)    
    dijkstras = all_dijkstras if all_dijkstras is not None else {sat.id: dijkstra(sats, i, sat.id, w_matrix) for sat in sats}

    res = 0
    for idx, sat in enumerate(sats):
        for sat_two in sats[idx+1:]:
            weight, nb_best_paths = dijkstras[sat.id][sat_two.id]
            if nb_best_paths == 0:
                continue
            res += weight

    n = len(sats)
    return res / (n*(n-1) / 2)

def closeness_centrality(sats: list[Satellite], i: int, sat_one_id, weight_mat = None, all_dijkstras = None):
    w_matrix = weight_mat if weight_mat is not None else weight_matrix(sats, i)
    sat_one = next((sat for sat in sats if sat.id == sat_one_id), None)
    
    dijkstras = all_dijkstras if all_dijkstras is not None else {sat.id: dijkstra(sats, i, sat.id, w_matrix) for sat in sats}

    res = 0
    for sat_two in sats:
        if sat_two.id != sat_one_id:
            weight_two_to_one, nb_best_paths_between_two_to_one = dijkstras[sat_two.id][sat_one.id]
            if nb_best_paths_between_two_to_one == 0:
                continue
            res += weight_two_to_one
    return (len(sats) - 1) / res

def betweeness_centrality(sats: list[Satellite], i: int, sat_one_id, weight_mat = None, all_dijkstras = None):
    # 2 -> 1 -> 3
    w_matrix = weight_mat if weight_mat is not None else weight_matrix(sats, i)
    sat_one = next((sat for sat in sats if sat.id == sat_one_id), None)
    
    dijkstras = all_dijkstras if all_dijkstras is not None else {sat.id: dijkstra(sats, i, sat.id, w_matrix) for sat in sats}

    res = 0
    for sat_two in sats:
        if sat_two.id != sat_one_id:
            weight_two_to_one, nb_best_paths_between_two_to_one = dijkstras[sat_two.id][sat_one.id]

            for sat_three in sats:
                if sat_three.id != sat_one_id and sat_three.id != sat_two.id:
                    weight_three_to_two, nb_best_paths_between_three_to_two = dijkstras[sat_three.id][sat_two.id]
                    weight_three_to_one, nb_best_paths_between_three_to_one = dijkstras[sat_three.id][sat_one.id]
                    if nb_best_paths_between_three_to_two == 0:
                        continue
                        
                    if (weight_two_to_one + weight_three_to_one) != weight_three_to_two:
                        nb_best_paths_between_two_to_three_passing_by_one = 0
                    else:
                        nb_best_paths_between_two_to_three_passing_by_one = nb_best_paths_between_two_to_one * nb_best_paths_between_three_to_one

                    res += nb_best_paths_between_two_to_three_passing_by_one / nb_best_paths_between_three_to_two
    return res

def isolate_subgraphs_dijkstra_method(sats: list[Satellite], i: int, weight_mat = None, all_dijkstras = None):
    w_matrix = weight_mat if weight_mat is not None else weight_matrix(sats, i)    
    dijkstras = all_dijkstras if all_dijkstras is not None else {sat.id: dijkstra(sats, i, sat.id, w_matrix) for sat in sats}
    sat_id_to_sat = {sat.id: sat for sat in sats}
    
    subgraphs = []
    candidates = set(sat.id for sat in sats)
    
    while candidates:
        subgraphs.append([])
        
        sat_id = candidates.pop()
        subgraphs[-1].append(sat_id_to_sat[sat_id])
        
        for candidate_left in list(candidates):
            if dijkstras[sat_id][candidate_left][1] != 0:
                subgraphs[-1].append(sat_id_to_sat[candidate_left])
                candidates.remove(candidate_left)
        
    return subgraphs

def isolate_subgraphs_bfs_method(sats: list[Satellite], i: int):    
    subgraphs = []
    
    visited = set()
    for sat in sats:
        if sat.id in visited:
            continue
            
        new_subgraph = []
        bfs_queue = deque([sat])
        
        while bfs_queue:
            current_sat = bfs_queue.popleft()
            
            visited.add(current_sat.id)
            new_subgraph.append(current_sat)
            
            for neighbor in current_sat.list_coordinates[i].neighbors:
                if neighbor.id not in visited:
                    bfs_queue.append(neighbor)
                    visited.add(neighbor.id) # Avoid getting pushed in queue multiple times
                    
        subgraphs.append(new_subgraph)
    
    return subgraphs

def isolate_subgraphs(sats: list[Satellite], i: int, weight_mat = None, all_dijkstras = None):
    if all_dijkstras is not None:
        return isolate_subgraphs_dijkstra_method(sats, i, weight_mat, all_dijkstras)
    else:
        return isolate_subgraphs_bfs_method(sats, i)

def modularity(sats: list[Satellite], i: int, classes_partition: dict[int, any], adjacency_mat = None, weight_mat = None, nb_unique_edges = None):
    adj_matrix = adjacency_mat if adjacency_mat is not None else adjacency_matrix(sats, i)
    w_matrix = weight_mat if weight_mat is not None else weight_matrix(sats, i)

    ids = [sat.id for sat in sats]
    classes_as_array = np.array([classes_partition[id] for id in ids])
    kronecker_delta = np.equal.outer(classes_as_array, classes_as_array).astype(int)
    
    two_m = adj_matrix.sum()
    A = adj_matrix.to_numpy()
    k = w_matrix.to_numpy().sum(axis=1)
    ki_times_kj_div_by_two_m = np.outer(k, k) / two_m 

    return np.sum((A - ki_times_kj_div_by_two_m) * kronecker_delta) / two_m