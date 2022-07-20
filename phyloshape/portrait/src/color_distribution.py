#!/usr/bin/env python
"""functions for profiling shape vertex_colors

"""

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