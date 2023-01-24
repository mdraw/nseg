import gunpowder as gp
from pathlib import Path

# voxel_size = gp.Coordinate((4, 4, 20))  # todo: estimate from data (4, 4, 20)
input_shape = gp.Coordinate((84, 84, 84))  # todo: increase (164, 164, 164)
output_shape = gp.Coordinate((44, 44, 44))  # (124, 124, 124)

voxel_size = gp.Coordinate((1,1,1))  # todo: estimate from data (4, 4, 20)
input_shape = gp.Coordinate((80, 80, 80))  # todo: increase (164, 164, 164)
output_shape = gp.Coordinate((80, 80, 80))  # (124, 124, 124)


# # For 2D CREMI
# voxel_size = gp.Coordinate((4, 4))
# input_shape = gp.Coordinate((164, 164))
# output_shape = gp.Coordinate((124, 124))



input_size = input_shape * voxel_size
output_size = output_shape * voxel_size

num_samples = 1  # number training cubes
batch_size = 5

out_dir = Path('./out')
out_dir.mkdir(exist_ok=True)
log_dir = out_dir / 'logs'
model_dir = out_dir / 'models'
