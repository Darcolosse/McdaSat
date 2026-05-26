from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterator
from .Point import Point
    

@dataclass
class InstantSatellite:
    """Represents the state of a satellite at a given instant t.

    Attributes:
        point: The position of the satellite at this instant t.
        neighbors: The list of neighboring satellite positions.
    """

    # Basic blueprint of an instant in time of a satellite
    point: Point
    _neighbors: dict[int, Satellite] = field(default_factory=dict)

    # For failure management
    _deactivated_neighbors: dict[int, Satellite] = field(default_factory=dict)

    @property
    def neighbors(self) -> Iterator[Satellite]:
        return self._neighbors.values()

    @property
    def neighbor_ids(self) -> Iterator[int]:
        return self._neighbors.keys()

    @property
    def neighbor_names(self) -> list[str]:
        return [sat.name for sat in self._neighbors.values()]

    def add_neighbor(self, neighbor: Satellite):
        self._neighbors[neighbor.id] = neighbor

    def has_neighbor(self, neighbor_id: int) -> bool:
        return neighbor_id in self._neighbors

    def deactivate_neighbors(self, neighbor_ids: list[int]):
        for neighbor_id in neighbor_ids:
            if neighbor_id in self._neighbors:
                self._deactivated_neighbors[neighbor_id] = self._neighbors.pop(neighbor_id)

    def reactivate_neighbors(self, neighbor_ids: list[int]):
        for neighbor_id in neighbor_ids:
            if neighbor_id in self._deactivated_neighbors:
                self._neighbors[neighbor_id] = self._deactivated_neighbors.pop(neighbor_id)

    def deactivate_all_neighbors(self):
        self._deactivated_neighbors.update(self._neighbors)
        self._neighbors.clear()

    def reactivate_all_neighbors(self):
        self._neighbors.update(self._deactivated_neighbors)
        self._deactivated_neighbors.clear()


@dataclass
class Satellite:
    """Represents a satellite and its trajectory over time.

    Attributes:
        name: The name of the satellite.
        list_coordinates: The list of recorded instants for this satellite.
    """

    id: int
    name: str
    list_coordinates: list[InstantSatellite] = field(default_factory=list)

    def add_instant(self, point: Point, neighbors: dict[str, Satellite] | None = None) -> None:
        """Record a new position instant for this satellite.

        Args:
            point: The position of the satellite at this instant.
            neighbors: The neighboring satellite positions. Defaults to empty list.
        """
        self.list_coordinates.append(InstantSatellite(point, neighbors or {}))