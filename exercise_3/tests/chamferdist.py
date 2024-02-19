#TODO: Implement Chamfer Distance Metric w/ G.T Mesh

# Read the GT meshes

# Read the generated meshes from the validation set

# Generate random points samples https://www.fwilliams.info/point-cloud-utils/sections/mesh_sampling/#generating-random-samples-on-a-mesh
# The paper used 30K points for distance measurement we also can use the same number OR since paper calculated SDFs with 16K points and we used 1/4 of it, we can use 10K points

# Calculate chamfer distance https://www.fwilliams.info/point-cloud-utils/sections/shape_metrics/

import argparse

import torch
import numpy as np

# pip install trimesh[all]
import trimesh

import point_cloud_utils as pcu

# https://github.com/otaheri/chamfer_distance
from chamfer_distance import ChamferDistance

def as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        assert len(scene_or_mesh.geometry) > 0
        mesh = trimesh.util.concatenate(
            tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                for g in scene_or_mesh.geometry.values()))
    else:
        assert isinstance(scene_or_mesh, trimesh.Trimesh)
        mesh = scene_or_mesh
    return mesh

def sample_mesh(m, n):
    vpos, _ = trimesh.sample.sample_surface(m, n)
    return torch.tensor(vpos, dtype=torch.float32, device="cuda")

''''
def get_chamfer_distance(mesh, ref_vertices, ref_faces, n):
    chamfer_dist = ChamferDistance()
    mesh = as_mesh(trimesh.load(mesh))
    ref = trimesh.Trimesh(vertices=ref_vertices, faces=ref_faces)

    # Sample mesh surfaces
    vpos_mesh = sample_mesh(mesh, n)
    vpos_ref = sample_mesh(ref, n)

    # Make sure l=1.0 maps to 1/10th of the AABB. https://arxiv.org/pdf/1612.00603.pdf
    scale = 10.0 / np.amax(np.amax(ref.vertices, axis=0) - np.amin(ref.vertices, axis=0))
    ref.vertices = ref.vertices * scale
    mesh.vertices = mesh.vertices * scale

    print(f"vpos_mesh: {vpos_mesh.shape}")
    dist1, dist2, idx1, idx2 = chamfer_dist(vpos_mesh[None, ...], vpos_ref[None, ...])
    loss = (torch.mean(dist1)) + (torch.mean(dist2))

    torch.cuda.synchronize()
    print(f"Loss: {loss}")


    #dist1, dist2 = chamfer_dist(vpos_mesh[None, ...], vpos_ref[None, ...])
    #loss = (torch.mean(dist1) + torch.mean(dist2)).item()
    
    #print("[%7d tris]: %1.5f" % (mesh.faces.shape[0], loss))

'''

def get_chamfer_distance(mesh_path_1, mesh_path_2):
    
    # p1 is an (n, 3)-shaped numpy array containing one point per row
    p1 = pcu.load_mesh_v(mesh_path_1)

    # p2 is an (m, 3)-shaped numpy array containing one point per row
    p2 = pcu.load_mesh_v(mesh_path_2)

    # Compute the chamfer distance between p1 and p2
    cd = pcu.chamfer_distance(p1, p2)

    print(f"Chamfer Distance: {cd}")
    return cd

def get_sinkhorn_distance(mesh_path_1, mesh_path_2):

    # p1 is an (n, 3)-shaped numpy array containing one point per row
    p1 = pcu.load_mesh_v(mesh_path_1)

    # p2 is an (m, 3)-shaped numpy array containing one point per row
    p2 = pcu.load_mesh_v(mesh_path_2)

    # Compute the chamfer distance between p1 and p2
    emd, pi = pcu.earth_movers_distance(p1, p2)

    return emd
