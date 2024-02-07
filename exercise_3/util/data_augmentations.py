import torch
from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotation, Scale


"""
    Fits the points into the unit sphere

    input:
    points: (n, 3) n: num_points, 3: xyz
    out:
    translated_and_scaled: same shape with points. Fitted into the unit sphere
"""
def scale_and_translate_to_unit_sphere(points):
    # Find the bounding box of the mesh
    min_coords, max_coords = torch.min(points, 0), torch.max(points, 0)

    # Calculate the center and size of the bounding box
    center = (min_coords + max_coords) / 2
    size = max(max_coords - min_coords)

    # Scale and translate mesh to fit within the unit sphere
    scaled = (1.0 / size) * points
    translated_and_scaled = scaled + (-center / size / 1.03)
    
    return translated_and_scaled



"""
    Applies a random rotation 

    input:
    points: (n, 3) n: num_points, 3: xyz
    out:
    rotated_pts: same shape with points. Fitted into the unit sphere
"""
def apply_random_rotation(points):
    rot = Rotate(R=random_rotation())
    rotated_pts = rot.transform_points(points) 
    return rotated_pts 



"""
    Applies a random nonuniform-scale. After, that it also fits to the unit sphere 

    input:
    points: (n, 3) n: num_points, 3: xyz
    scale_factor: max scale we can get in a dimension
    out:
    scaled_pts: same shape with points. Fitted into the unit sphere
"""
def apply_random_nonuniform_scale(points, scale_factor):
    scale = Scale(x=scale_factor * torch.rand(1), y=scale_factor * torch.rand(1), z=scale_factor * torch.rand(1))
    scaled_pts = scale.transform_points(points) 
    # Fit into the unit sphere
    scaled_pts = scale_and_translate_to_unit_sphere(scaled_pts)
    return scaled_pts 

