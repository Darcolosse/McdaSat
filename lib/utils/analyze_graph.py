from ..classes import Satellite
from concurrent.futures import ProcessPoolExecutor
import time
import numpy as np

def run_for_each_range_each_instant(data, satellites, func, *args, **kwargs):
    start = time.perf_counter()
    res = {key: [func(satellites[key], i, *args, **kwargs) for i in range(data.shape[1])] for key in satellites.keys()}
    end = time.perf_counter()
    print(f"Temps écoulé : {end - start :.2f} secondes\n")
    return res

def run_for_each_range_each_instant_on_calculated_data(data, satellites, calculated_data, func, *args, **kwargs):
    start = time.perf_counter()
    res = {key: [func(satellites[key], i, calculated_data[key], *args, **kwargs) for i in range(data.shape[1])] for key in satellites.keys()}
    end = time.perf_counter()
    print(f"Temps écoulé : {end - start :.2f} secondes\n")
    return res

def count_isolated_subgraphs(sats: list[Satellite], i: int, precalculated_subgraphs_instants: list[list[Satellite]]):
    return len(precalculated_subgraphs_instants[i])

def count_isolated_clusters(sats: list[Satellite], i: int, precalculated_subgraphs_instants: list[list[Satellite]]):
    lengths = np.array([len(subgraph) for subgraph in precalculated_subgraphs_instants[i]])
    return np.sum(lengths > 1)