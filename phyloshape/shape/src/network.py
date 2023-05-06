#!/usr/bin/env python

"""Core PhyloShape class object of the phyloshape package.

"""

from __future__ import annotations
from typing import Mapping, Set, NamedTuple
# import random

from loguru import logger
import numpy as np
from phyloshape.utils import COORD_TYPE, ID_TYPE
from numpy.typing import ArrayLike
from typing import Union, List, Dict
from copy import deepcopy
from collections import OrderedDict

logger = logger.bind(name="phyloshape")


class Edge(NamedTuple):
    """A single edge between two vertices representing a length."""
    start: int
    end: int
    dist: float

    def __repr__(self) -> str:
        return f"Edge({self.start}>{self.end}:{self.dist:.5g})"


class Path(list):
    """A list of one or more Edges creating a path of a given length."""
    def __init__(self, *edges: Edge, idx: int = 0):
        super().__init__(edges)
        self.idx = idx
        """Index of this path in its managed list."""

    def __repr__(self) -> str:
        return f"Path({', '.join(str(i) for i in self)})"

    @property
    def dist(self) -> float:
        return sum(i.dist for i in self)

    @property
    def target(self) -> int:
        return self[-1]


class PathManager(list):
    """Class for comparing and operating on multiple Path objects.

    This is a list-like object of filled with Path objects sorted by
    lowest dist, or if same dist, then by lowest target ID. Each path
    in the list must have a unique target ID or an exception will be
    raised. 

    This is used to merge paths which extend to the same terminal
    vertex, creating a tree. Multiple target vertices can have the
    same ..?
    """
    def __init__(self, *paths: Path):
        super().__init__(sorted(paths, key=lambda x: (x.dist, x.target)))
        self.target_to_pidx = {p.target: i for i, p in enumerate(self.paths)}
        """A dict of {vertex: index in paths} for ActivePaths in .paths"""
        assert len(self.paths) == len(self.target_to_pid), (
            "PathManager cannot accept repeated target ids!")

    def get_path_by_target(self, target: int) -> None:
        ...

    def get_path_to(self, vertex_id: int) -> Path:
        """Return ActivePath ending at vertex_id"""
        return self[self.target_to_pidx[vertex_id]]

    def get_rank(self, path: Path, start: int = 0, step: int = 1) -> int:
        """Return the rank of a path.

        Iteratively compare this ActivePath to all others starting at
        position start, and moving forward [or backward] by step. Once
        the path is shorter than the one it is comparing to the index
        of that position is returned.
        """
        # other path to compare to initially
        opath = self.paths[start]

        # loop while path is longer than comparison path
        while (
            path._len > opath._len
            or path._len == opath._len and path[-1] > opath[-1]
        ):
            # increment the ste
            start += step
            if start < len(self.paths):
                opath = self.paths[start]
            else:
                break
        logger.debug(f"rank of {path} = {start}")
        return start

    def pop_shortest(self) -> Path:
        """..."""
        # subtract 1 from the index of all remaining vertices.
        for vertex in self.target_to_pid:
            self.target_to_pid[vertex] -= 1
        # return the first element (shortest) in paths
        return self.paths.pop(0)

    def _insert(self, at_pidx: int, path: Path) -> None:
        """Insert an ActivePath at a specific index. Called by .add()"""
        self.paths.insert(at_pidx, path)
        for pid in range(at_pidx, len(self.paths)):
            path = self.paths[pid]
            self.target_to_pid[path[-1]] += 1
        self.target_to_pid[path[-1]] = at_pidx

    def add(self, path: Path) -> None:
        """...

        Parameters
        ----------
        path: ActivePath
            A candidate ActivePath for ...
        """
        target = path[-1]

        # if no paths exist then add candidate to path list
        if not self.paths:
            self.paths.append(path)
            self.target_to_pid[target] = 0

        # some paths already exist and end at the same target
        elif target in self.target_to_pid:
            # get the current path ending at this target
            defending_pid = self.target_to_pid[target]
            defending_path = self.paths[defending_pid]

            # if candidate path is shorter then get rank and insert
            cond1 = path._len < defending_path._len
            cond2 = path._len == defending_path._len and path < defending_path
            if cond1 or cond2:

                # find rank of candidate path search backwards from defending
                new_pid = self.get_rank(path, start=defending_pid, step=-1)

                # remove old path with same target at index=defending_pid
                logger.info(f"remove path with same target {self.paths[defending_pid]} at pid={defending_pid}")
                del self.paths[defending_pid]

                # update ranks
                for change_pid in range(new_pid, defending_pid - 1):
                    ppp = self.target_to_pid[self.paths[change_pid][-1]]
                    self.target_to_pid[self.paths[change_pid][-1]] += 1
                    logger.info(f"updating {self.paths[change_pid][-1]} from rank {ppp} to {ppp + 1}")

                # set index 
                self.target_to_pid[target] = new_pid

                self.paths.insert(new_pid, path)
        else:
            logger.warning("FROM HERE")
            self._insert(self.get_rank(path=path), path)


class Network:
    """Network object with connections among vertices and their lengths.

    Parameters
    ----------
    pairs: np.ndaray
        Array of shape (npairs, 2) with start and end IDs of each edge
    edge_lens: Series[float]
        Series of floats for the length of each edge.
    """
    def __init__(self, pairs: np.ndarray, edge_lens: np.ndarray):

        # Non-init attributes
        self.adjacency: Mapping[int, Dict[int, float]] = {}
        """: Mapping of edges as {vertex: {vertex: length}, ...}"""
        self.id_set: Set[int] = set()
        """: Set of all vertices in adjacency."""
        self.shortest_path: float = {}
        """: ..."""

        # fill the non-init attributes with info in the init data
        pairs = np.array([]) if pairs is None else np.array(pairs, dtype=ID_TYPE)
        if edge_lens is None:
            edge_lens = np.repeat(1, pairs.shape[0])
        else:
            edge_lens = np.array(edge_lens, dtype=COORD_TYPE)

        # nedges must match n edge lens
        assert pairs.shape[0] == edge_lens.shape[0]

        # fill adjacency to that self.adjacency[x, y] has its length
        for (id_1, id_2), elen in zip(pairs, edge_lens):

            # ...
            if id_1 not in self.adjacency:
                self.adjacency[id_1] = {}
            if id_2 not in self.adjacency:
                self.adjacency[id_2] = {}
            self.adjacency[id_2][id_1] = self.adjacency[id_1][id_2] = elen

            # ...
            # self.shortest_path[(id_1, id_2)] = ((id_1, id_2), elen)
            # self.shortest_path[(id_2, id_1)] = ((id_2, id_1), elen)

        self.id_set = set(self.adjacency)

    def __getitem__(self, vertex_id: int) -> Dict[int, float]:
        """Return {vertex_id: edge_len, ...} for each edge from vertex_id"""
        return self.adjacency[vertex_id]

    def __bool__(self) -> bool:
        """Return True if any edges exist in the network"""
        return bool(self.adjacency)

    def get_shortest_path_from(
        self,
        vertex_id: int,
        cutoff: float = np.inf,
        cache_res: bool = False,
    ) -> List:
        """Return shortest path(s) from vertex_id to another vertex.

        Only returns paths that reach another vertex within the cutoff
        distance, found using Dijkstra's algorithm.

        Note
        ----
        Currently used in `ColorProfile` for shape-associated color
        distribution measurements.

        Parameters
        ----------
        vertex_id:
            ...
        cutoff:
            ...
        cache_res:
            If True then .shortest_path dict is updated...
        """
        assert vertex_id in self.id_set

        # list to fill with found vertices
        found_to_return = []

        # set of all other vertices from which far verts will be dropped
        pending_ids = self.id_set - {vertex_id}

        # build stack of all paths out from this vertex within cutoff
        stack_len_increasing = []
        for next_id in self[vertex_id]:
            dist = self.adjacency[vertex_id][next_id]
            if dist <= cutoff:
                path = ActivePath((vertex_id, next_id))
                path._len = dist
                stack_len_increasing.append(path)

        # Create PathManager with functions to compare these paths
        active_paths = PathManager(stack_len_increasing)

        # iterate over paths popping and merging until just shortest remains
        while active_paths:

            # get the shortest path
            path = active_paths.pop_shortest()
            logger.info(f"{path}")

            # get the vertex at end of this path
            target = path[-1]

            # create an info dict for this path
            forward_info = {
                "path": path,
                "len": path._len,
                "to_id": target,
            }

            # [optionally] cache this info dict to be reused later
            if cache_res and (vertex_id, target_v_id) not in self.__shortest_path:
                # store path in both directions
                self.__shortest_path[(vertex_id, target_v_id)] = deepcopy(forward_info)
                self.__shortest_path[(target_v_id, vertex_id)] = {
                    "path": shortest_active_p.vertices()[::-1],
                    "len": shortest_active_p.dist(),
                    "to_id": vertex_id,
                }

            # store info and remove target from search possibilities
            found_to_return.append(forward_info)
            pending_ids.remove(target)

            # iterate over connected vertices
            for vidx, elen in self[target].items():

                # if vertex is among those we are searching towards
                if vidx in pending_ids:  # not in shortest_active_p:

                    # add vertex and its len to current path
                    path.append(vidx)
                    path._len += elen

                    # and store this new longer path (back) to PathManager
                    if path._len <= cutoff:
                        active_paths.add(path)

        # return the shortest ActivePath from vertex_id to ...
        return found_to_return


    def find_unions(self) -> None:
        """Return the union of X...

        """
        unions = []
        candidate_vs = set(self.adjacency)
        while candidate_vs:
            new_root = candidate_vs.pop()
            unions.append({new_root})
            waiting_vs = set([next_v for next_v in self.adjacency[new_root] if next_v in candidate_vs])
            while candidate_vs and waiting_vs:
                next_v = waiting_vs.pop()
                unions[-1].add(next_v)
                candidate_vs.discard(next_v)
                for next_v in self.adjacency[next_v]:
                    if next_v in candidate_vs:
                        waiting_vs.add(next_v)
        return unions


#####################################################################
#####################################################################
#####################################################################

class IdNetwork:
    """

    """
    def __init__(
        self,
        pairs: Union[ArrayLike, List, None] = None,
        edge_lens: Union[ArrayLike, List, None] = None,
    ):
        """
        :param pairs: the ids of two ends of edges
        :param edge_lens: the lengths of edges
        """
        self.__pairs = np.array([], dtype=ID_TYPE) if pairs is None \
            else np.array(pairs, dtype=ID_TYPE)
        self.__edge_lens = np.array([1.] * len(self.__pairs), dtype=COORD_TYPE) if edge_lens is None \
            else np.array(edge_lens, dtype=COORD_TYPE)
        assert len(self.__pairs) == len(self.__edge_lens)
        self.adjacency = {}
        self.__id_set = set()
        self.__shortest_path = {}  # TODO use disk space to store if too large
        for (id_1, id_2), this_len in zip(self.__pairs, self.__edge_lens):
            self.__id_set.add(id_1)
            self.__id_set.add(id_2)
            if id_1 not in self.adjacency:
                self.adjacency[id_1] = OrderedDict()
            if id_2 not in self.adjacency:
                self.adjacency[id_2] = OrderedDict()
            self.adjacency[id_2][id_1] = self.adjacency[id_1][id_2] = this_len
            self.__shortest_path[(id_1, id_2)] = {"path": [id_1, id_2], "len": this_len, "to_id": id_2}
            self.__shortest_path[(id_2, id_1)] = {"path": [id_2, id_1], "len": this_len, "to_id": id_1}

    def __getitem__(self, vertex_id: int) -> Dict[int, float]:
        """Return dict w/ {vertex_id: edge_len, ...} for each edge from this vertex."""
        return self.adjacency[vertex_id]

    def __bool__(self) -> bool:
        """Return True if any edges exist in the network"""
        return bool(self.adjacency)

    def find_shortest_paths_from(self,
                                 vertex_id: int,
                                 cutoff: float = float("inf"),
                                 cache_res: bool = False) -> List:
        """for a given distance cutoff, find all neighboring points along the 3D model using Dijkstra's algorithm

        :param vertex_id:
        :param cutoff:
        :param cache_res:
        :return:
        """
        assert vertex_id in self.__id_set
        ### initialization
        found_to_return = []
        pending_ids = self.__id_set - {vertex_id}
        stack_len_increasing = []
        for next_id in self[vertex_id]:
            path_info = self.__shortest_path[(vertex_id, next_id)]  # A1 was used
            if path_info["len"] <= cutoff:
                stack_len_increasing.append(_ActivePath(path_info["path"], path_info["len"]))
                # found_to_return.append(deepcopy(path_info))
        active_paths = _ActivePaths(stack_len_increasing)
        ### searching
        while active_paths:
            # print("- round -")
            # for a_p in active_paths:
            #     print(a_p)
            shortest_active_p = active_paths.pop_shortest()
            target_v_id = shortest_active_p.target_v_id()
            forward_info = {"path": shortest_active_p.vertices(), "len": shortest_active_p.dist(), "to_id": target_v_id}
            if cache_res and (vertex_id, target_v_id) not in self.__shortest_path:
                self.__shortest_path[(vertex_id, target_v_id)] = deepcopy(forward_info)
                self.__shortest_path[(target_v_id, vertex_id)] = \
                    {"path": shortest_active_p.vertices()[::-1], "len": shortest_active_p.dist(), "to_id": vertex_id}
            found_to_return.append(forward_info)
            pending_ids.remove(target_v_id)
            for next_id, add_len in self[target_v_id].items():
                if next_id in pending_ids:  # not in shortest_active_p:
                    proposed_path = shortest_active_p.appended(vertex_id=next_id, add_len=add_len)
                    if proposed_path.dist() <= cutoff:
                        active_paths.add(proposed_path)
        return found_to_return

    def find_unions(self):
        unions = []
        candidate_vs = set(self.adjacency)
        while candidate_vs:
            new_root = candidate_vs.pop()
            unions.append({new_root})
            waiting_vs = set([next_v for next_v in self.adjacency[new_root] if next_v in candidate_vs])
            while candidate_vs and waiting_vs:
                next_v = waiting_vs.pop()
                unions[-1].add(next_v)
                candidate_vs.discard(next_v)
                for next_v in self.adjacency[next_v]:
                    if next_v in candidate_vs:
                        waiting_vs.add(next_v)
        return unions


class _ActivePaths:
    """A class to manage active path collections.

    Paths leading to the same target should be (are?) automatically
    merged into the shorted one.
    """
    def __init__(self, active_path_list: List = []):
        # sort by length and target_v_id
        self.__active_paths = sorted(active_path_list, key=lambda x: (x.dist(), x.target_v_id()))
        self.__target_to_pid = {p.target_v_id(): p_id for p_id, p in enumerate(self.__active_paths)}
        assert len(self.__target_to_pid) == len(self.__active_paths), "Initialization failed: repeated target ids!"

    def __bool__(self):
        return bool(self.__active_paths)

    def __getitem__(self, path_id):
        return self.__active_paths[path_id]

    def __iter__(self):
        for act_p in self.__active_paths:
            yield act_p

    def __len__(self):
        return len(self.__active_paths)

    def __insert(self, at_p_id, active_path):
        """
        only to be called by self.add(), which has a duplicate checking and sorting
        :param active_path:
        :param at_p_id:
        :return:
        """
        n_paths = len(self.__active_paths)
        assert 0 <= at_p_id <= n_paths
        for change_id in range(at_p_id, n_paths):
            self.__target_to_pid[self.__active_paths[change_id].target_v_id()] += 1
        self.__target_to_pid[active_path.target_v_id()] = at_p_id
        self.__active_paths.insert(at_p_id, active_path)

    def __find_rank(self,
                    active_path,
                    start_from: int = 0,
                    step: int = 1):
        """
        :param active_path:
        :param start_from: path_id
        :param step: 1 or -1
        :return:
        """
        where_to_insert = start_from
        this_len = active_path.dist()
        compare_len = self.__active_paths[where_to_insert].dist()
        while this_len > compare_len or \
                (this_len == compare_len and
                 active_path.target_v_id() > self.__active_paths[where_to_insert].target_v_id()):
            where_to_insert += step
            if where_to_insert < len(self.__active_paths):
                compare_len = self.__active_paths[where_to_insert].dist()
            else:
                break
        # print("  find rank for " + str(active_path) + ": " + str(where_to_insert))
        return where_to_insert

    def get_path_to(self, target_v_id):
        return self.__active_paths[self.__target_to_pid[target_v_id]]

    def add(self, candidate_act_path):
        target_v_id = candidate_act_path.target_v_id()
        if not self.__active_paths:
            self.__active_paths.append(candidate_act_path)
            self.__target_to_pid[target_v_id] = 0
        elif target_v_id in self.__target_to_pid:
            defending_pid = self.__target_to_pid[target_v_id]
            defending_path = self.__active_paths[defending_pid]
            if candidate_act_path.dist() < defending_path.dist() or \
                    (candidate_act_path.dist() == defending_path.dist() and
                     candidate_act_path.vertices() < defending_path.vertices()):
                new_pid = self.__find_rank(candidate_act_path, start_from=defending_pid, step=-1)
                for change_pid in range(new_pid, defending_pid - 1):
                    self.__target_to_pid[self.__active_paths[change_pid].target_v_id()] += 1  # rank these back
                self.__target_to_pid[target_v_id] = new_pid
                del self.__active_paths[defending_pid]
                self.__active_paths.insert(new_pid, candidate_act_path)
            else:
                pass
        else:
            # print("  from here")
            self.__insert(self.__find_rank(active_path=candidate_act_path), candidate_act_path)

    def pop_shortest(self):
        for target_v_id in self.__target_to_pid:
            self.__target_to_pid[target_v_id] -= 1
        return self.__active_paths.pop(0)

    def delete(self, p_id: int):
        # check p_id
        n_paths = len(self.__active_paths)
        if p_id % n_paths == n_paths - 1:
            pass
        elif p_id % n_paths == 0:
            for target_v_id in self.__target_to_pid:
                self.__target_to_pid[target_v_id] -= 1
        else:
            for change_pid in range(1 + p_id % n_paths, n_paths):
                self.__target_to_pid[self.__active_paths[change_pid].target_v_id()] -= 1
        del self.__active_paths[p_id]

    # def pop(self, p_id: int = -1):
    #     # check p_id
    #     n_paths = len(self.__active_paths)
    #     if p_id % n_paths == n_paths - 1:
    #         pass
    #     elif p_id % n_paths == 0:
    #         for target_v_id in self.__target_to_pid:
    #             self.__target_to_pid[target_v_id] -= 1
    #     else:
    #         for change_id in range(1 + p_id % n_paths, n_paths):
    #             self.__target_to_pid[self.__active_paths[change_id].target_v_id()] -= 1
    #     return self.__active_paths.pop(p_id)


class _ActivePath:
    """A non-repeating path between a set of vertices."""
    def __init__(self, vertex_list: List = [], initial_len: int = 0):
        self._vertex_list = vertex_list
        self._vertex_set = set(vertex_list)
        self._len = initial_len
        assert len(self._vertex_set) == len(self._vertex_list), (
            "Initialization failed: repeated vertex in a path!")

    def __str__(self):
        return "->".join((str(v) for v in self._vertex_list)) + ":" + str(self._len)

    def __repr__(self):
        return str(self)

    def __contains__(self, vertex_id: int) -> bool:
        return vertex_id in self._vertex_set

    def __len__(self) -> int:
        return len(self._vertex_list)

    def __deepcopy__(self, memodict={}) -> _ActivePath:
        """Return a deepcopy

        This custom deepcopy function is used because...
        """
        new_active_path = _ActivePath()
        new_active_path._vertex_list = list(self._vertex_list)
        new_active_path._vertex_set = set(self._vertex_set)
        new_active_path._len = self._len
        return new_active_path

    # def append(self, vertex_id, add_len):
    #     if vertex_id in self:
    #         raise ValueError(vertex_id + " existed in current path!")
    #     else:
    #         self._vertex_set.add(vertex_id)
    #         self._vertex_list.append(vertex_id)
    #         self._len += add_len

    def appended(self, vertex_id: int, add_len: float) -> _ActivePath:
        """...

        """
        if vertex_id in self:
            raise ValueError(f"{vertex_id} exists in current path!")
        else:
            new_active_path = deepcopy(self)
            new_active_path._vertex_set.add(vertex_id)
            new_active_path._vertex_list.append(vertex_id)
            new_active_path._len += add_len
            return new_active_path

    # def append_quick(self, vertex_id: int, add_len: int):
    #     self._vertex_set.add(vertex_id)
    #     self._vertex_list.append(vertex_id)
    #     self._len += add_len

    def vertices(self) -> List[int]:
        """Return a deepcopy of the vertex list."""
        return deepcopy(self._vertex_list)

    def target_v_id(self) -> int:
        """Return the vertex at the end of the path (last vertex in list)"""
        return self._vertex_list[-1]

    def dist(self) -> float:
        """
        physical length of the path, instead of len(self) or len(self._vertex_list)
        :return:
        """
        return self._len


def get_random_network(v_size: int = 100):
    """Generate a random network and ..."""
    n_edge = int(v_size * 1.2)
    return (
        np.random.choice(range(v_size), size=(n_edge, 2)),
        np.random.random(n_edge),
    )


if __name__ == "__main__":

    PAIRS, EDGE_LENS = get_random_network()

    # VSIZE = 10
    # NEDGES = int(VSIZE * 1.2)
    # pairs = np.random.choice(range(VSIZE), size=(NEDGES, 2))
    # edge_lens = np.random.random(NEDGES)
    # PAIRS = [
    #     (1, 2),
    #     (2, 3),
    #     (3, 4),
    #     (1, 4),
    # ]
    # EDGE_LENS = [0.5, 0.5, 0.5, 1.5]

    # net = Network(PAIRS, EDGE_LENS)
    # # print(net.adjacency)

    # p0 = ActivePath([1, 2])
    # p1 = ActivePath([2, 3])
    # p2 = ActivePath([3, 4])
    # p3 = ActivePath([2, 4])

    # m = PathManager([p0, p1, p2])

    # logger.info(m)
    # # print(m.target_to_pid)
    # # print(m.get_path_to(2))
    # # print(m.get_rank(p0))

    # m.add(p3)
    # net.get_shortest_path_from(2)


    net = IdNetwork(PAIRS, EDGE_LENS)
    shortest = net.find_shortest_paths_from(3, cutoff=0.5)
    logger.warning(f"shortest: {shortest}")
    unions = net.find_unions()
    for u in unions:
        logger.info(f"union: {u}")