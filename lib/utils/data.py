from pathlib import Path
from ..classes.Satellite import Satellite
from ..classes.Point import Point

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import time

def load_data(filepath: str | Path, separator: str = ",", header: int | None = None) -> pd.DataFrame:
    """Load raw data from a CSV file.

    Args:
        filepath: Path to the CSV file (str or Path object).
        separator: Column delimiter used in the CSV file. Defaults to ','.
        header: Row number to use as column names.
                None means no header row. Defaults to None.

    Returns:
        Raw DataFrame loaded from the CSV file.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
        ValueError: If the file extension is not '.csv'.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: '{filepath}'")

    if filepath.suffix.lower() != ".csv":
        raise ValueError(
            f"Expected a '.csv' file, got '{filepath.suffix}' instead."
        )

    return pd.read_csv(filepath, sep=separator, header=header)

def struct_data(df: pd.DataFrame, static_range: int = 28000) -> list[Satellite]:
    print(f"Chargement des satellites imposant une portée statique uniforme de {static_range}m ({static_range/1000}km)")

    # Create columns label
    df.columns = [f"t{i}" for i in range(len(df.columns))]

    satellites = []

    start = time.perf_counter()   

    # Satellite coordinate data processing
    for i in range(0, len(df), 3):
        coords = df.iloc[i:i+3].values
        satellite = Satellite(i, f"SAT-{int(i/3):03d}")

        for j in range(coords.shape[1]):
            satellite.add_instant(Point(coords[0, j], coords[1, j], coords[2, j]))

        satellites.append(satellite)

    # Satellite neighbors processing
    for index, sat in enumerate(satellites):
        for other_sat in satellites[index+1:]:
            for i in range(len(sat.list_coordinates)):
                in_range = sat.list_coordinates[i].point.in_range( other_sat.list_coordinates[i].point, static_range)
                if in_range:
                    sat.list_coordinates[i].add_neighbor(other_sat)
                    other_sat.list_coordinates[i].add_neighbor(sat)
                    
    end = time.perf_counter()

    print(f"{len(satellites)} satellites ont été chargés")
    print(f"Temps écoulé : {end - start :.2f} secondes\n")

    return satellites

def simulate_failure(satellites: list[Satellite], ids: list[int], instants: list[int]):
    for i in instants:
        for sat in satellites:
            if sat.id in ids:
                sat.list_coordinates[i].deactivate_all_neighbors()
            else:
                sat.list_coordinates[i].deactivate_neighbors(ids)

def undo_failure(satellites: list[Satellite], ids: list[int], instants: list[int]):
    for i in instants:
        for sat in satellites:
            if sat.id in ids:
                sat.list_coordinates[i].reactivate_all_neighbors()
            else:
                sat.list_coordinates[i].reactivate_neighbors(ids)

def show_data(satellites: list[Satellite], number: int) -> None:
    plt.figure(figsize=(15, 15))
    axes = plt.axes(projection="3d")

    for sat in satellites[:number]:
        xs = np.array([instant.point.x for instant in sat.list_coordinates])
        ys = np.array([instant.point.y for instant in sat.list_coordinates])
        zs = np.array([instant.point.z for instant in sat.list_coordinates])

        axes.plot(xs, ys, zs, label=f"Sat {sat.name}")

    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Z')
    axes.set_title("Trajectoires des Satellites")
    plt.show()


def show_instant_nodes(satellites: list[Satellite], instant: int) -> None:
    plt.show()
    plt.figure(figsize=(15, 15))
    axes = plt.axes(projection="3d")

    # Create Nodes
    for sat in satellites:
        point = sat.list_coordinates[instant].point    
        axes.scatter(
            xs=point.x,
            ys=point.y,
            zs=point.z,
            s=50,
            c='blue',
            label='Satellites')

    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Z')
    axes.set_title(f"t{instant} des Satellites")
    # plt.legend()
    plt.show()

def show_instant_nodes_edges(satellites: list[Satellite], instant: int) -> None:
    plt.figure(figsize=(15, 15))
    axes = plt.axes(projection="3d")

    # Create Nodes and edges
    for sat in satellites:
        # Nodes
        point = sat.list_coordinates[instant].point    
        axes.scatter(
            xs=point.x,
            ys=point.y,
            zs=point.z,
            s=50,
            #c='blue',
            label='Satellites')

        # Edges
        for neighbor in sat.list_coordinates[instant].neighbors:
            neighbor_point = neighbor.list_coordinates[instant].point
            axes.plot(
                [point.x, neighbor_point.x],
                [point.y, neighbor_point.y],
                [point.z, neighbor_point.z],
                'r-',
                alpha=0.1,
                linewidth=1
            )


    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Z')
    axes.set_title(f"t{instant} des Satellites")
    # plt.legend()
    plt.show()

def show_instant_nodes_neighbor(satellites: list[Satellite], instant: int, label: bool = True) -> None:
    fig = plt.figure(figsize=(15, 15))
    axes = plt.axes(projection="3d")


    densities = np.array([
        len(sat.list_coordinates[instant].neighbors)
        for sat in satellites
    ])

    norm = plt.Normalize(vmin=densities.min(), vmax=densities.max())
    colormap = cm.viridis

    # Create Nodes and edges
    for sat, density in zip(satellites, densities):
        # Nodes
        color = (255/255, 0/255, 0/255, 1.0) if density == 0 else colormap(norm(density)) 
        point = sat.list_coordinates[instant].point
        axes.scatter(
            xs=point.x,
            ys=point.y,
            zs=point.z,
            s=50,
            color=color,
            label='Satellites')
        
        if label:
            axes.text(
                x=point.x,
                y=point.y,
                z=point.z,
                s=sat.name,
                fontsize=8,
                color='black',
                ha='right',
                va='bottom')

        # Edges
        for neighbor in sat.list_coordinates[instant].neighbors:
            neighbor_point = neighbor.list_coordinates[instant].point
            axes.plot(
                [point.x, neighbor_point.x],
                [point.y, neighbor_point.y],
                [point.z, neighbor_point.z],
                'r-',
                alpha=0.1,
                linewidth=1
            )

    scalar_mappable = cm.ScalarMappable(norm=norm, cmap=colormap)
    scalar_mappable.set_array(densities)
    fig.colorbar(scalar_mappable, ax=axes, label="Nombre de voisins", shrink=0.5)

    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Z')
    axes.set_title(f"t{instant} des Satellites")
    # plt.legend()
    plt.show()
