import numpy as np
from PIL import Image
import colorsys 

class WavefrontParser():
    """This parser parses lines from .obj files."""
    def __init__(self, filepath):
        """Read file directory of the .obj files."""
        self.file = open(filepath)
        self.lines = self.file.readlines()
        
    def get_dataframe(self, tex_image_path):
        """Create a numpy array to store xyz vertex coordinates， uv texture coordinates, RGB values and HSV values"""
        # call the functions
        vertex_coords = self.parser_vertex_coords()
        tex_coords = self.parser_tex_coords()
        faces = self.parser_faces()
        
        # create an empty numpy array
        df = np.zeros(shape=(len(vertex_coords), 11))
        
        # load texture image
        img =  Image.open(tex_image_path)
        # obtain image dimension
        width, height = img.size
        
        
        # fill in the array row by row
        for row in range(len(df)):
            # add xyz and uv values
            v = int((list(faces.keys())[row])) 
            vt = int(faces[v])
            v_list = vertex_coords[v-1] # index starts with 1, we need to minus 1
            vt_list = tex_coords[vt-1]
            
            # get rgb and hsv values
            pixel_x = int(vt_list[0]*width)
            pixel_y = int(vt_list[1]*height)
            rgb = list(img.getpixel((pixel_x,pixel_y)))
            std_rgb = [i / 255 for i in rgb]
            std_hsv = list(colorsys.rgb_to_hsv(std_rgb[0],std_rgb[1],std_rgb[2]))
            
            df[row] = v_list + vt_list + std_rgb + std_hsv
        
        return df    
    
    def parser_vertex_coords(self):
        """Parse file line by line and store vertex coordinates"""
        # create lists and dictionary to store accordingly information
        vertex_coords = []
        
        # read file line by line
        for line in self.lines:
            line = line.strip().split(' ') 
            if line[0] == 'v':
                vertex_coords.append([float(i) for i in line[1:4]])
        
        return vertex_coords
    
    def parser_vertex_color(self):
        """Parse file line by line and store vertex color"""
        # create lists and dictionary to store accordingly information
        vertex_color = []
        
        # read file line by line
        for line in self.lines:
            line = line.strip().split(' ') 
            if line[0] == 'v' and len(line) == 7:
                vertex_color.append([float(i) for i in line[4:]])
            if line[0] == 'v' and len(line) == 4:
                print("Sorry, no vertex color is included. Please check your .obj file")
                break
        
        return vertex_color

    def parser_tex_coords(self):
        """Parse file line by line and store texture coordinates"""
        # create lists and dictionary to store accordingly information
        tex_coords = []
        
        # read file line by line
        for line in self.lines:
            line = line.strip().split(' ') 
            if line[0] == 'vt':
                tex_coords.append([float(i) for i in line[1:3]])
        
        return tex_coords
    
    def parser_faces(self):
        """Parse file line by line and store faces (vertex coordinates/texture coordinates)"""
        # create lists and dictionary to store accordingly information
        faces = {}
        
        # read file line by line
        for line in self.lines:
            line = line.strip().split(' ') 
            if line[0] == 'f':
                f = line[1:]
                for i in range(0,len(f)):
                    f[i] = f[i].split('/')
                    faces[float((f[i])[0])] = float((f[i])[1]) #v is the key, vt is the element
        
        return faces