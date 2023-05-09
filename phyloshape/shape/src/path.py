#!/usr/bin/env python

"""Path class to represent connections in a Graph.

These objects are used in the Graph class (see graph.py).

Classes
--------
Edge:
    stores start ID, end ID, dist for an edge.
Path:
    List of Edge to represent a path over a 3D model.
PathExtender:
    Container for Paths ending at the same target Vertex ID. This is
    used inside of the Graph object to find the shortest path between
    two vertices, or all shortest paths from a vertex to others.
"""

from __future__ import annotations
from typing import Set
from loguru import logger


logger = logger.bind(name="phyloshape")


class Edge:
    """A single edge between two vertices representing a length.

    This is simpler than a Vector object, used only for representing
    edges between Vertex IDs in a graph for computing distances.
    """
    def __init__(self, start: int, end: int, dist: float):
        self.start = start
        self.end = end
        self.dist = dist

    def __repr__(self) -> str:
        return f"Edge({self.start}>{self.end}:{self.dist:.5g})"


class Path(list):
    """A list of ordered Edges that together create a longer Path.

    Two paths can be compared to find which is shorter path to a target.
    Path objects are intended for internal use only.
    """
    def __init__(self, *edges: Edge):
        super().__init__(edges)

    def __repr__(self) -> str:
        path = ">".join(str(i.start) for i in self)
        return f"Path({path}>{self.target}:{self.dist:.5g})"

    # total_ordering will expand to create all comparisons ---------
    def __eq__(self, other: Path) -> bool:
        return (
            (self.dist, self.target) ==
            (other.dist, other.target)
        )

    def __lt__(self, other: Path) -> bool:
        return (
            (self.dist, self.target) <
            (other.dist, other.target)
        )

    def __le__(self, other: Path) -> bool:
        return (
            (self.dist, self.target) <=
            (other.dist, other.target)
        )

    def __gt__(self, other: Path) -> bool:
        return (
            (self.dist, self.target) >
            (other.dist, other.target)
        )

    def __ge__(self, other: Path) -> bool:
        return (
            (self.dist, self.target) >=
            (other.dist, other.target)
        )
    # --------------------------------------------------------------

    def __copy__(self) -> Path:
        return Path(*self)

    @property
    def dist(self) -> float:
        return sum(i.dist for i in self)

    @property
    def target(self) -> int:
        return self[-1].end

    @property
    def start(self) -> int:
        return self[0].start

    @property
    def vertices(self) -> Set[int]:
        return {i.start for i in self} | {i.end for i in self}


class PathExtender(list):
    """Class for comparing and extending Paths. Internal use.

    This is a list-like object of filled with sorted Path objects.
    Each path in the list must have a unique target ID or an exception
    will be raised.
    """
    def __init__(self, *paths: Path):
        super().__init__(sorted(paths, key=lambda x: (x.dist, x.target)))

        # also index paths by their target
        self.target_to_pid = {p.target: i for i, p in enumerate(self)}

        assert len(self) == len(self.target_to_pid), (
            "PathExtender cannot accept repeated target ids!")

    def get_path_to(self, vertex_id: int) -> Path:
        """Return Path ending at vertex_id"""
        return self[self.target_to_pid[vertex_id]]

    def get_rank(self, path: Path, start: int = 0, step: int = 1) -> int:
        """Return the rank of a path.

        Iteratively compare this Path to all others starting at start
        and moving forward [or backward] by step. Once the path is
        shorter than the one it is comparing to return the index.
        """
        # other path to compare to initially
        opath = self[start]

        # loop while path is longer than comparison path
        while path > opath:
            # increment the step
            start += step
            if start < len(self):
                opath = self[start]
            else:
                break
        return start

    def pop_shortest(self) -> Path:
        """Return the shortest path and remove it from path list"""
        # subtract 1 from the index of all remaining vertices.
        for vertex in self.target_to_pid:
            self.target_to_pid[vertex] -= 1
        # return the first element (shortest) in paths
        return self.pop(0)

    def _insert(self, at_pidx: int, path: Path) -> None:
        """Insert an ActivePath at a specific index. Called by .add()"""
        self.insert(at_pidx, path)
        self.target_to_pid[path.target] = at_pidx
        for pid in range(at_pidx + 1, len(self)):
            path = self[pid]
            self.target_to_pid[path.target] += 1

    def add(self, path: Path) -> None:
        """Add a Path to .paths list and update target indices.

        If an existing Path ends in the same

        Parameters
        ----------
        path: Path
            A candidate Path for ...
        """
        target = path.target

        # if no paths exist then add candidate as first in path list
        if not self:
            self.append(path)
            self.target_to_pid[target] = 0
            logger.debug(f"adding first {path}")
            return

        # if no other existing path ends in the same target insert it
        if target not in self.target_to_pid:
            self._insert(self.get_rank(path=path), path)
            logger.debug(f"adding {path} to novel target")
            return

        # replace current path that ends at the same target
        cpid = self.target_to_pid[target]
        cpath = self[cpid]
        logger.debug(f"testing candidate {path} to replace {cpath}")

        # if candidate path is shorter then get rank and insert
        while path < cpath:

            # get rank of candidate path (backward from current path to target)
            new_pid = self.get_rank(path, start=cpid, step=-1)

            # remove old path with same target
            logger.info(f"rm path w/ same target {cpath} at pid={cpid}")
            self.pop(cpid)

            # update ranks of paths between new and old
            for pid in range(new_pid, cpid - 1):
                self.target_to_pid[self[pid].target] += 1

            # set new path to target at new_pid
            self.target_to_pid[target] = new_pid
            self._insert(new_pid, path)
            logger.info(f"adding {path} to replace {cpath}")
        else:
            logger.info(f"rejecting {path} to replace {cpath}")


if __name__ == "__main__":

    import phyloshape
    phyloshape.set_log_level("TRACE")

    e0 = Edge(0, 1, 0.1)
    e1 = Edge(1, 2, 0.2)
    e2 = Edge(2, 3, 0.3)

    p0 = Path(e0, e1)
    p1 = Path(e1, e2)
    pm = PathExtender(p0, p1)
