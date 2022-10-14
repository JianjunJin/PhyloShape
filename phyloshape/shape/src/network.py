#!/usr/bin/env python

"""Core PhyloShape class object of the phyloshape package.

"""
import random

from loguru import logger
import numpy as np
from phyloshape.utils import COORD_TYPE, ID_TYPE
from numpy.typing import ArrayLike
from typing import Union, List, Dict
from copy import deepcopy
from collections import OrderedDict
from loguru import logger
logger = logger.bind(name="phyloshape")


class IdNetwork:
    def __init__(self,
                 pairs: Union[ArrayLike, List, None] = None,
                 edge_lens: Union[ArrayLike, List, None] = None):
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

    def __getitem__(self, vertex_id) -> Dict:
        # assert vertex_id in self.__id_set
        return self.adjacency[vertex_id]

    def __bool__(self):
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
        """

        :return:
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


class _ActivePaths:
    """
    A class to manage active path collections.
    Paths leading to the same target should be automatically merged into the shorted one.
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
    """
    Create a class to manage an active path. Vertices in a Path should be non-repeatable.
    """
    def __init__(self, vertex_list: List = [], initial_len: int = 0):
        self._vertex_list = vertex_list
        self._vertex_set = set(vertex_list)
        self._len = initial_len
        assert len(self._vertex_set) == len(self._vertex_list), "Initialization failed: repeated vertex in a path!"

    def __str__(self):
        return "->".join([str(v_) for v_ in self._vertex_list]) + ":" + str(self._len)

    def __repr__(self):
        return "->".join([str(v_) for v_ in self._vertex_list]) + ":" + str(self._len)

    def __contains__(self, vertex_id) -> bool:
        return vertex_id in self._vertex_set

    def __len__(self) -> int:
        return len(self._vertex_list)

    def __deepcopy__(self, memodict={}):
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

    def appended(self, vertex_id, add_len):
        if vertex_id in self:
            raise ValueError(str(vertex_id) + " existed in current path!")
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

    def vertices(self) -> List:
        return deepcopy(self._vertex_list)

    def target_v_id(self) -> int:
        return self._vertex_list[-1]

    def dist(self) -> float:
        """
        physical length of the path, instead of len(self) or len(self._vertex_list)
        :return:
        """
        return self._len


def test_net():
    # this_net = IdNetwork(pairs=[[1, 2],
    #                             [1, 3],
    #                             [2, 4],
    #                             [3, 5],
    #                             [3, 6],
    #                             [4, 6],
    #                             [5, 6],
    #                             [7, 8],
    #                             [7, 9],
    #                             [9, 10],
    #                             [11, 12]],
    #                      edge_lens=[0.9, 0.1, 0.95, 0.2, 0.2, 0.2, 0.2, 0.5, 0.15, 0.4, 0.6])
    v_size = 10000
    n_edge = int(v_size * 1.2)
    this_net = IdNetwork(pairs=np.random.choice(range(v_size), size=(n_edge, 2)),
                         edge_lens=np.random.random(n_edge))
    res_paths = this_net.find_shortest_paths_from(1, cutoff=0.5)
    return res_paths

