#!/usr/bin/env python
"""classes and functions for profiling shape colors

"""
from phyloshape.shape.src.shape import Shape
from phyloshape.utils import RGB_TYPE, OP_RGB_TYPE, rgb_to_hsv, rgb_to_hex
from typing import Union, List, Dict
from loguru import logger
import numpy as np
logger = logger.bind(name="phyloshape")

from phyloshape.utils.src.process import ProgressLogger, ProgressText


class ColorProfile:
    def __init__(self, shape: Shape):
        self.shape = shape

    # def color_component_across_vertices(self):
    #     self.shape.vertices.colors

    def color_variation_across_vertices(self,
                                        dist_values: List[float],
                                        n_start_vertices: int = 1000,
                                        user_defined_vertices: List[int] = []):
        res_var_dict = {dist_group_: [] for dist_group_ in dist_values}
        shape = self.shape
        max_id = len(shape.vertices) - 1
        sim_num = n_start_vertices - len(user_defined_vertices)
        chosen_ids = list(np.random.randint(low=0, high=max_id, size=sim_num)) + user_defined_vertices
        dist_val_sort = sorted(dist_values, reverse=True)
        logger.info("searching")
        progress_1 = ProgressLogger(n_start_vertices)
        progress_1a = ProgressText(n_start_vertices)
        for v_id in chosen_ids:
            # cutoff = max(dist_val_sort)
            path_info_list = shape.network.find_shortest_paths_from(v_id, cutoff=dist_val_sort[0])
            id_group_per_dist_range = {dist_group_: [] for dist_group_ in dist_val_sort}
            for p_info in path_info_list:
                for go_d, dist_upper_bd in enumerate(dist_val_sort[:-1]): # the largest dist may contain the most points
                    dist_lower_bd = dist_val_sort[go_d + 1]
                    if p_info["len"] > dist_lower_bd:
                        id_group_per_dist_range[dist_upper_bd].append(p_info["to_id"])
                        break
                else:
                    id_group_per_dist_range[dist_val_sort[-1]].append(p_info["to_id"])

            this_color = shape.vertices.colors[v_id]
            for go_d, dist_group in enumerate(dist_val_sort):
                neighbor_ids = []
                for add_dist_g in dist_val_sort[go_d:]:
                    neighbor_ids += id_group_per_dist_range[add_dist_g]
                #TODO: temporarily use vertex color, use texture later
                neighboring_colors = shape.vertices.colors[neighbor_ids]
                color_vars = abs(np.array(neighboring_colors, dtype=OP_RGB_TYPE) - this_color)
                res_var_dict[dist_group].append(np.array(np.max(color_vars, axis=0), dtype=RGB_TYPE))
            progress_1.update()
            progress_1a.update()

        logger.info("summarizing")
        progress_2 = ProgressLogger(len(dist_values))
        progress_2a = ProgressText(len(dist_values))
        for dist_group, res_var in res_var_dict.items():
            res_var_dict[dist_group] = np.array(res_var, dtype=RGB_TYPE)
            progress_2.update()
            progress_2a.update()

        return res_var_dict








# Yue's original code
# def get_dataframe(self, tex_image_path):
#     """Create a numpy array to store xyz vertices coordinates， uv texture coordinates, RGB values and HSV values"""
#     # call the functions
#     vertex_coords = self.__parser_vertex_coords()
#     tex_coords = self.__parser_tex_coords()
#     faces = self.__parser_faces()
#
#     # create an empty numpy array
#     df = np.zeros(shape=(len(vertex_coords), 11))
#
#     # load texture image
#     img = Image.open(tex_image_path)
#     # obtain image dimension
#     width, height = img.size
#
#     # fill in the array row by row
#     for row in range(len(df)):
#         # add xyz and uv values
#         v = int((list(faces.keys())[row]))
#         vt = int(faces[v])
#         v_list = vertex_coords[v - 1]  # index starts with 1, we need to minus 1
#         vt_list = tex_coords[vt - 1]
#
#         # get rgb and hsv values
#         pixel_x = int(vt_list[0] * width)
#         pixel_y = int(vt_list[1] * height)
#         rgb = list(img.getpixel((pixel_x, pixel_y)))
#         hsv = rgb_to_hsv(rgb)
#
#         df[row] = v_list + vt_list + rgb + hsv
#
#     return df