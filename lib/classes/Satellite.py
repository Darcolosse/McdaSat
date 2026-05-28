from __future__ import annotations
from dataclasses import dataclass, field
from .Point import Point

EMISSION_RANGES = [20_000, 40_000, 60_000]


@dataclass
class InstantSatellite:
    """Represents the state of a satellite at a given instant t.

    Attributes:
        point: The position of the satellite at this instant t.
        neighbors: Neighboring satellites, selected by crescendo range (20/40/60 km).
    """

    point: Point
    # Cumulative neighbors per range level: range_m -> {sat_id -> Satellite}
    _neighbors_by_range: dict[int, dict[int, Satellite]] = field(default_factory=dict)
    _deactivated_ids: set[int] = field(default_factory=set)

    @property
    def neighbors(self) -> list[Satellite]:
        """Return neighbors at the lowest emission range that has at least one active neighbor."""
        for r in EMISSION_RANGES:
            level = [v for k, v in self._neighbors_by_range.get(r, {}).items()
                     if k not in self._deactivated_ids]
            if level:
                return level
        return []

    @property
    def neighbor_ids(self) -> list[int]:
        return [s.id for s in self.neighbors]

    @property
    def neighbor_names(self) -> list[str]:
        return [s.name for s in self.neighbors]

    def add_neighbor(self, neighbor: Satellite, range_m: int):
        if range_m not in self._neighbors_by_range:
            self._neighbors_by_range[range_m] = {}
        self._neighbors_by_range[range_m][neighbor.id] = neighbor

    def has_neighbor(self, neighbor_id: int) -> bool:
        return any(neighbor_id in d for d in self._neighbors_by_range.values())

    def deactivate_neighbors(self, neighbor_ids: list[int]):
        self._deactivated_ids.update(neighbor_ids)

    def reactivate_neighbors(self, neighbor_ids: list[int]):
        self._deactivated_ids.difference_update(neighbor_ids)

    def deactivate_all_neighbors(self):
        all_ids = {k for d in self._neighbors_by_range.values() for k in d}
        self._deactivated_ids.update(all_ids)

    def reactivate_all_neighbors(self):
        self._deactivated_ids.clear()


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

    def add_instant(self, point: Point) -> None:
        """Record a new position instant for this satellite."""
        self.list_coordinates.append(InstantSatellite(point))