from __future__ import annotations


from dataclasses import dataclass, field

from Point import Point


@dataclass
class InstantSatellite:
    """Represents the state of a satellite at a given instant t.

    Attributes:
        point: The position of the satellite at this instant t.
        neighbors: The list of neighboring satellite positions.
    """

    point: Point
    neighbors: list[Satellite] = field(default_factory=list)


@dataclass
class Satellite:
    """Represents a satellite and its trajectory over time.

    Attributes:
        name: The name of the satellite.
        list_coordinates: The list of recorded instants for this satellite.
    """

    name: str
    list_coordinates: list[InstantSatellite] = field(default_factory=list)

    def add_instant(self, point: Point, neighbors: list[Point] | None = None) -> None:
        """Record a new position instant for this satellite.

        Args:
            point: The position of the satellite at this instant.
            neighbors: The neighboring satellite positions. Defaults to empty list.
        """
        self.list_coordinates.append(InstantSatellite(point, neighbors or []))