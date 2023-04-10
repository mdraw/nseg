import gunpowder as gp
from pathlib import Path

# input_shape = gp.Coordinate((84, 84, 84))  # todo: increase (164, 164, 164)
# output_shape = gp.Coordinate((44, 44, 44))  # (124, 124, 124)


# zebrafinch: zyx space
# sizes (voxel_size, input_size, ...) need to be in zyx!


# voxel_size = gp.Coordinate((9, 9, 20))  # todo: estimate from data (4, 4, 20)
# voxel_size = gp.Coordinate((9, 9, 20))  # todo: estimate from data (4, 4, 20)
# voxel_size = gp.Coordinate((1, 1, 1))  # todo: estimate from data (4, 4, 20)
# input_shape = gp.Coordinate((60, 90, 90))  # todo: increase (164, 164, 164)
# output_shape = input_shape
# input_shape = gp.Coordinate((60, 90, 90))  # todo: increase (164, 164, 164)
# output_shape = gp.Coordinate((60, 90, 90))  # (124, 124, 124)
# input_shape = gp.Coordinate((180, 180, 180))  # todo: increase (164, 164, 164)
# output_shape = gp.Coordinate((180, 180, 180))  # (124, 124, 124)



# input_shape = gp.Coordinate((84, 84, 84)  # todo: increase (164, 164, 164)
#                             )
# output_shape = gp.Coordinate((44, 44, 44)  # (124, 124, 124)
#                              )
# output_shape = gp.Coordinate((84, 84, 84)  # (124, 124, 124)
#                              )

voxel_size = gp.Coordinate((20, 9, 9))  # zyx

padding = 'valid'

# input size in voxels

if padding == 'same':
    input_shape = gp.Coordinate((80, 80, 80))  # todo: increase (164, 164, 164)
    output_shape = input_shape
elif padding == 'valid':
    # s = 164
    # input_shape = gp.Coordinate((s, s, s))
    input_shape = gp.Coordinate((84, 268, 268))
    offset = gp.Coordinate((40, 40, 40))
    output_shape = input_shape - offset

    inference_input_shape = gp.Coordinate((96, 484, 484))
    inference_output_shape = inference_input_shape - offset


input_size = input_shape * voxel_size
output_size = output_shape * voxel_size

inference_input_size = inference_input_shape * voxel_size
inference_output_size = inference_output_shape * voxel_size


out_dir = Path('./out')
out_dir.mkdir(exist_ok=True)
log_dir = out_dir / 'logs'
model_dir = out_dir / 'models'


"""
voxel_size = gp.Coordinate((20, 9, 9))
input_shape = gp.Coordinate((1, 2, 3))
-->
request: (18, 27, 20)
       = (9 * 2, 9 * 3, 20 * 1)

;

voxel_size = gp.Coordinate((20, 7, 9))
input_shape = gp.Coordinate((1, 2, 3))
-->
request: (20, 14, 27)
       = (1 * 20, 2 * 7, 3 * 9)


"""
