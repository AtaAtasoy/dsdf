import point_cloud_utils as pcu

#TODO: Load the mesh
#v, f, n = pcu.load_mesh_vfn("mesh.obj")

num_samples = 30000 # 30K points used for chamfer 500 for earthmover
#num_samples = 500 # 500 points for chamfer dist

# Generates barycentric coordinates
fid, bc = pcu.sample_mesh_random(v, f, num_samples)


rand_positions = pcu.interpolate_barycentric_coords(f, fid, bc, v)

# export the positions into a ply
# Make sure that it follows: p1 is an (n, 3)-shaped numpy array containing one point per row
