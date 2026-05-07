from __future__ import annotations
import math

from dataclasses import dataclass
from typing import Self

@dataclass
class Point:
    """Represents a point in 3D space.

    Attributes:
        x: The x-coordinate of the point.
        y: The y-coordinate of the point.
        z: The z-coordinate of the point.
    """
    x: float
    y: float
    z: float

    def in_range(self, point: Self, radius: float) -> bool:
        """Check if a given point is within a specified radius.

        Args:
            point: The target point to check.
            radius: The maximum allowed distance.

        Returns:
            True if the point is within the radius, False otherwise.
        """
        dist_x = point.x - self.x
        dist_y = point.y - self.y
        dist_z = point.z - self.z

        if (abs(dist_x) > radius) or (abs(dist_y) > radius) or  (abs(dist_z) > radius):
            return False
        else:
            # Normally math.sqrt(dist_x**2 + dist_y**2 + dist_z**2), but sqrt is omitted
            # for optimization by comparing squared distance against radius**2 instead.
            distance = (dist_x**2 + dist_y**2 + dist_z**2)
            return distance <= radius**2

    def distance(self, point: Self) -> float:
        """Compute the Euclidean distance between this point and another.

        Args:
            point: The target point.

        Returns:
            The distance as a float.
        """
        dist_x = point.x - self.x
        dist_y = point.y - self.y
        dist_z = point.z - self.z

        return math.sqrt(dist_x**2 + dist_y**2 + dist_z**2)