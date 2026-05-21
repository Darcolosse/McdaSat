import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
import math

def show_edge(satellites: list[Satellite]) -> None:
    plt.figure(figsize=(15, 15))
    axes = plt.axes(projection="3d")

    # Create Nodes
    for sat in satellites:
        point = sat.list_coordinates[0].point    
        axes.scatter(
            x=point.x,
            y=point.y,
            z=point.z,
            s=50,
            c='blue',
            label='Satellites')

        for neighbor in sat.list_coordinates[0].neighbors:
            neighbor_point = neighbor.list_coordinates[0].point
            axes.plot(
                [point.x, neighbor_point.x],
                [point.y, neighbor_point.y],
                [point.z, neighbor_point.z],
                'r-',
                alpha=0.03,
                linewidth=1
            )


    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Z')
    axes.set_title("t0 des Satellites")
    # plt.legend()
    plt.show()


def show_edge(satellites: list[Satellite]) -> None:
    plt.figure(figsize=(15, 15))
    axes = plt.axes(projection="3d")

    # Stockage des positions des satellites
    satellite_points = []
    satellite_names = []

    for sat in satellites:
        if not sat.list_coordinates:
            continue

        point = sat.list_coordinates[0].point
        satellite_points.append((point.x, point.y, point.z))
        satellite_names.append(sat.name)

    # Convertir en tableaux NumPy
    satellite_points = np.array(satellite_points)

    # Tracer les satellites
    if len(satellite_points) > 0:
        axes.scatter(
            satellite_points[:, 0],
            satellite_points[:, 1],
            satellite_points[:, 2],
            s=50,
            c='blue',
            label='Satellites'
        )

    # Tracer les arêtes avec une boucle
    for sat in satellites:
        if not sat.list_coordinates:
            continue

        point = sat.list_coordinates[0].point
        for neighbor in sat.list_coordinates[0].neighbors:
            if not neighbor.list_coordinates:
                continue

            neighbor_point = neighbor.list_coordinates[0].point
            axes.plot(
                [point.x, neighbor_point.x],
                [point.y, neighbor_point.y],
                [point.z, neighbor_point.z],
                'r-',
                alpha=0.03,
                linewidth=1
            )

    # Ajouter les étiquettes
    for i, name in enumerate(satellite_names):
        axes.text(
            satellite_points[i, 0],
            satellite_points[i, 1],
            satellite_points[i, 2],
            name,
            fontsize=8,
            ha='center',
            va='center'
        )

    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Z')
    axes.set_title("t0 des Satellites")
    plt.legend()
    plt.show()
    
    
    