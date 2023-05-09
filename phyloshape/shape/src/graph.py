#!/usr/bin/env python

"""Graph class object for constructing paths between vertices on a model.

"""

from typing import List, Set, Iterator, Optional, Tuple
from copy import copy
from loguru import logger
import numpy as np
from phyloshape.shape.src.path import Edge, Path, PathExtender

logger = logger.bind(name="phyloshape")


class Graph(dict):
    """Graph class represents all edges between vertices.

    This class can be used to analyze the connections to find the
    shortest path between vertices, or to cluster into disconnected
    clusters of vertices.
    """
    def __init__(self, edges: np.ndarray, dists: np.ndarray):

        # self.edges: Mapping[int, Path] = {}
        for (start, end), dist in zip(edges, dists):
            if start not in self:
                self[start] = [Edge(start, end, dist)]
            else:
                self[start].append(Edge(start, end, dist))
            if end not in self:
                self[end] = [Edge(end, start, dist)]
            else:
                self[end].append(Edge(end, start, dist))

    ###################################################################
    # user-facing functions
    ###################################################################
    def get_nearest_neighbor_path(self, start: int) -> Path:
        """Return shortest path to a neighbor

        If multiple vertices are the same distance this will return only
        the first one, based on the lowest vertex ID.
        """
        for path in self._iter_paths_from(start):
            return path

    def get_shortest_path(self, start: int, end: int) -> Path:
        """Return the shortest path between two vertices"""
        for path in self._iter_paths_from(start):
            if path.target == end:
                return path

    def get_all_paths_from(self, start: int, cutoff: float = np.inf) -> Path:
        """Return all outgoing paths from a vertex

        The paths can be limited to a cutoff distance to create a radius
        search to return only paths to vertices within this distance.
        """
        return list(self._iter_paths_from(start, cutoff))

    ###################################################################
    # helper functions
    ###################################################################
    def _get_out_paths(self, start: int, cutoff: float = np.inf) -> PathExtender:
        """Return a list of Paths starting from the start vertex.

        The returned PathExtender is a superclass of List with functions
        to compare and extend Paths. It is used in _iter_paths_from().
        """
        stack = []
        for edge in self[start]:
            if edge.dist <= cutoff:
                stack.append(Path(edge))
        return PathExtender(*stack)

    def _iter_paths_from(
        self,
        start: int,
        cutoff: float = np.inf,
        id_set: Optional[Set[int]] = None,
    ) -> Iterator[Path]:
        """Return shortest path(s) from vertex_id to all other Nodes.

        Only returns paths that reach another vertex within the cutoff
        distance using Dijkstra's algorithm. See get_all_paths_from().
        Creates Path objects from Edge data in Graph, extends Paths,
        and yields them.

        Parameters
        ----------
        id_set: Set[int] or None
            Set of targets to find paths to. If None then all vertices
            other than start are targets.
        """
        # set of candidate vertices to find a path to
        if id_set is None:
            id_set = set(self)

        # start vertex cannot be present or path might reverse/loop
        id_set.discard(start)

        # debugging
        # logger.debug(f"searching paths from v={start} to {len(id_set)} vertices")

        # yield directly connected vertices if in target id set
        pathlist = self._get_out_paths(start, cutoff)
        for path in pathlist:
            if path.target in id_set:
                yield copy(path)

        # yield longer paths to remaining vertices by extending paths
        while pathlist:

            # get the shortest path
            path = pathlist.pop_shortest()

            # get the vertex at end of this path
            target = path.target

            # store info and remove target from search possibilities
            id_set.remove(target)

            # iterate over connected vertices
            for edge in self[target]:

                # if vertex is among those we are searching towards
                if edge.end in id_set:

                    # path back to PathManager to contiue to extend
                    if path.dist + edge.dist <= cutoff:

                        # extend path by edge (target, vidx) to current path
                        path.append(edge)
                        yield copy(path)

                        # add this longer path back to Manager
                        pathlist.add(path)

    def _get_vertex_clusters(self) -> List[Set[int]]:
        """Return clusters of connected vertices.

        The largest cluster is typically the true 3D model, while some
        other small artifacts are sometimes present and can be excluded
        after clustering by shared edges.
        """
        # list to store sets...
        unions = []

        # start with all vertices and ...
        verts = set(self)
        while verts:

            # seed a new cluster
            seed = verts.pop()
            unions.append({seed})

            # get extensions from the seed
            connected = set([e.end for e in self[seed] if e.end in verts])

            # some vertices remain and some extensions are possible
            while verts and connected:
                next_v = connected.pop()
                unions[-1].add(next_v)
                verts.discard(next_v)
                for edge in self[next_v]:
                    if edge.end in verts:
                        connected.add(edge.end)
        return sorted(unions, key=len, reverse=True)

    def _iter_path_traversal(self, start: int) -> Iterator[Tuple[int, int]]:
        """Yield (id0, idx1) tuples of ordered edges visiting every vertex"""
        # start with all vertices
        id_set = set(self)

        # iteratively find path to nearest neighbor of remaining IDs
        while 1:
            path = next(self._iter_paths_from(start, id_set=id_set))
            yield (path.start, path.target)
            start = path.target
            id_set.remove(path.target)
            if not id_set:
                break


if __name__ == "__main__":

    import phyloshape
    phyloshape.set_log_level("TRACE")

    e0 = Edge(0, 1, 0.1)
    e1 = Edge(1, 2, 0.2)
    e2 = Edge(2, 3, 0.3)

    p0 = Path(e0, e1)
    p1 = Path(e1, e2)
    pm = PathExtender(p0, p1)

    PAIRS = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (1, 4),
    ]
    EDGE_LENS = [0.2, 0.5, 0.5, 0.5, 0.9]

    g = Graph(PAIRS, EDGE_LENS)
    # print(g.get_all_paths_from(1))
    # print(g.get_shortest_path(1, 4))
    # print(g._get_vertex_clusters())
    # p = g.get_nearest_neighbor_path(0)
    # print(p)
    # p = g.get_nearest_neighbor_path(p.target)
    # print(p)

    print(list(g._iter_path_traversal(0)))

    # # VSIZE = 10
    # # NEDGES = int(VSIZE * 1.2)
    # # pairs = np.random.choice(range(VSIZE), size=(NEDGES, 2))
    # # edge_lens = np.random.random(NEDGES)
    # PAIRS = [
    #     (1, 2),
    #     (2, 3),
    #     (3, 4),
    #     (1, 4),
    # ]
    # EDGE_LENS = [0.5, 0.5, 0.5, 0.5]

    # # net = Network(PAIRS, EDGE_LENS)
    # # print(net.adjacency)

    # e0 = Edge(1, 2, 0.3)
    # e1 = Edge(2, 3, 0.5)
    # e2 = Edge(3, 4, 0.5)
    # e3 = Edge(1, 4, 0.5)

    # p0 = Path(e0)
    # p1 = Path(e1)
    # p2 = Path(e2)
    # p3 = Path(e3)

    # # print(p2 == p3)
    # # print(p0)
    # # p0.append(Edge(4, 5, 0.1))
    # # print(p0)
    # # print(copy(p0))

    # # m = PathManager(p0, p1, p2)
    # # print(m.target_to_pid)
    # # m.add(p3)
    # # print(m.target_to_pid)
    # # print(m.get_path_to(4))

    # net = Network(PAIRS, EDGE_LENS)
    # print(net.get_vertex_clusters())
    # print(net)
    # print(sorted(get_all_paths_from(net, 1)))
    # print(net.get_shortest_path(1, 4))

    # net = IdNetwork(PAIRS, EDGE_LENS)
    # shortest = net.find_shortest_paths_from(3, cutoff=0.5)
    # logger.warning(f"shortest: {shortest}")
    # unions = net.find_unions()
    # for u in unions:
    #     logger.info(f"union: {u}")
