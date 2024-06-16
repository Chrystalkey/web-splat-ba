import numpy
from plyfile import PlyData, PlyElement
import random as r
from array import array

properties = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0",
        "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]

 
def generate_header():
    global properties, vertices
    plyheader = f"ply\nformat ascii 1.0\nelement vertex {len(vertices)}\n"
    for p in properties:
        plyheader = f"{plyheader}property float {p}\n"
    return f"{plyheader}end_header\n"

def rgb2sh(rgb):
    return (rgb - 0.5) / 0.28209479177387814

def sh2rgb(sh):
    return sh * 0.28209479177387814 + 0.5

def build_record(position:str, colour:(float, float, float), opacity:str = None, scale:str = None, rot:str = None):
    opacity = opacity if opacity is not None else "1"
    colour = (rgb2sh(colour[0]/255.), rgb2sh(colour[1]/255.), rgb2sh(colour[2]/255.))
    colour = f"{colour[0]} {colour[1]} {colour[2]}"
    rot = "1 0 0 0" if rot is None else rot
    return f"{position} 1 0 0 {colour} {opacity} {scale} {rot}\n"

vertices = [
        build_record("1 1 0", (255., 0., 0.), scale="5 5 0"),
        build_record("2 2 0", (0., 255., 0.), scale="5 5 0"),       
        build_record("3 3 0", (0., 0., 255.), scale="5 5 0"),
       ]

with open("generated.ply", "w") as plyfile:
    plyfile.write(generate_header())
    for ver in vertices:
        plyfile.write(ver)
