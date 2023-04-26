# Based on readme: https://github.com/seung-lab/zmesh

from zmesh import Mesher

labels = ... # some dense volumetric labeled image
mesher = Mesher( (4,4,40) ) # anisotropy of image

# initial marching cubes pass
# close controls whether meshes touching
# the image boundary are left open or closed
mesher.mesh(labels, close=False)

meshes = []
for obj_id in mesher.ids():
  meshes.append(
    mesher.get_mesh(
      obj_id,
      normals=False, # whether to calculate normals or not

      # tries to reduce triangles by this factor
      # 0 disables simplification
      simplification_factor=100,

      # Max tolerable error in physical distance
      max_simplification_error=8,
      # whether meshes should be centered in the voxel
      # on (0,0,0) [False] or (0.5,0.5,0.5) [True]
      voxel_centered=False,
    )
  )
  mesher.erase(obj_id) # delete high res mesh

mesher.clear() # clear memory retained by mesher

mesh = meshes[0]
mesh = mesher.simplify(
  mesh,
  # same as simplification_factor in get_mesh
  reduction_factor=100,
  # same as max_simplification_error in get_mesh
  max_error=40,
  compute_normals=False, # whether to also compute face normals
) # apply simplifier to a pre-existing mesh

# compute normals without simplifying
mesh = mesher.compute_normals(mesh)

mesh.vertices
mesh.faces
mesh.normals
mesh.triangles() # compute triangles from vertices and faces

# Extremely common obj format
with open('iconic_doge.obj', 'wb') as f:
  f.write(mesh.to_obj())

# Common binary format
with open('iconic_doge.ply', 'wb') as f:
  f.write(mesh.to_ply())

# Neuroglancer Precomputed format
with open('10001001:0', 'wb') as f:
  f.write(mesh.to_precomputed())