import numpy as np

def remove_points_outside_of_interior(xs, ys, zs, model):
    pts = filter(lambda pt : model.is_inside(*pt), zip(xs, ys, zs))
    xs,ys,zs = zip(*pts)
    return np.array(xs), np.array(ys), np.array(zs)
    
def generate_random_scatterers_in_box(x_min, x_max, y_min, y_max, z_min, z_max, num_scatterers, thickness):
    xs = np.random.uniform(low=x_min-thickness, high=x_max+thickness, size=(num_scatterers,))
    ys = np.random.uniform(low=y_min-thickness, high=y_max+thickness, size=(num_scatterers,))
    zs = np.random.uniform(low=z_min-thickness, high=z_max+thickness, size=(num_scatterers,))
    return xs, ys, zs

def point_inside_polygon(x, y, poly):
    """
    Determine if points are inside a given polygon or not.
    Returns true or false.
    """
    from matplotlib.path import Path
    num_verts = len(poly)
    codes = [Path.MOVETO]+[Path.LINETO]*(num_verts-1)+[Path.CLOSEPOLY]
    verts = poly+[poly[0]] # dummy closing vertex
    assert len(verts) == len(codes)
    
    path = Path(verts, codes)
    pts = np.array([[x,y]])
    return path.contains_points( pts )[0]
    