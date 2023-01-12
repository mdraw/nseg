import gunpowder as gp

voxel_size = gp.Coordinate((1, 1, 1)  # todo: estimate from data (4, 4, 20)
                           )

input_shape = gp.Coordinate((84, 84, 84)  # todo: increase (164, 164, 164)
                            )
output_shape = gp.Coordinate((44, 44, 44)  # (124, 124, 124)
                             )

input_size = input_shape * voxel_size
output_size = output_shape * voxel_size

num_samples = 4  # number training cubes
batch_size = 5
