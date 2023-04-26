# Originally from train_mtlsd.py

from matplotlib import pyplot as plt

from nseg.shared import create_lut

lsd_channels = {
    'offset (y)': 0,
    'offset (x)': 1,
    'offset (z)': 2,
    'orient (y)': 3,
    'orient (x)': 4,
    'orient (z)': 5,
    'yx change': 6,  # todo: order correct?
    'yz change': 7,
    'xz change': 8,
    'voxel count': 9
}

aff_channels = {
    'affs_0': 0,  # todo: fix names
    'affs_1': 1,
    'affs_2': 2,
    # 'affs_3': 3,
    # 'affs_4': 4,
    # 'affs_5': 5,
}


def imshow(
        tb, it,
        raw=None,
        ground_truth=None,
        target=None,
        prediction=None,
        h=None,
        shader='jet',
        subplot=True,
        channel=0,
        target_name='target',
        prediction_name='prediction'):
    raw = raw[:, :, :, raw.shape[-1] // 2] if raw is not None else None
    ground_truth = ground_truth[:, :, :, ground_truth.shape[-1] // 2] if ground_truth is not None else None
    target = target[:, :, :, :, target.shape[-1] // 2] if target is not None else None
    prediction = prediction[:, :, :, :, prediction.shape[-1] // 2] if prediction is not None else None

    rows = 0

    if raw is not None:
        rows += 1
        cols = raw.shape[0] if len(raw.shape) > 2 else 1
    if ground_truth is not None:
        rows += 1
        cols = ground_truth.shape[0] if len(ground_truth.shape) > 2 else 1
    if target is not None:
        rows += 1
        cols = target.shape[0] if len(target.shape) > 2 else 1
    if prediction is not None:
        rows += 1
        cols = prediction.shape[0] if len(prediction.shape) > 2 else 1

    if subplot:
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(10, 4),
            sharex=True,
            sharey=True,
            squeeze=False)

    if h is not None:
        fig.subplots_adjust(hspace=h)

    def wrapper(data, row, name="raw"):

        if subplot:
            if len(data.shape) == 2:
                if name == 'raw':
                    axes[0][0].imshow(data, cmap='gray')
                    axes[0][0].set_title(name)
                else:
                    axes[row][0].imshow(create_lut(data))
                    axes[row][0].set_title(name)

            elif len(data.shape) == 3:
                for i, im in enumerate(data):
                    if name == 'raw':
                        axes[0][i].imshow(im, cmap='gray')
                        axes[0][i].set_title(name)
                    else:
                        axes[row][i].imshow(create_lut(im))
                        axes[row][i].set_title(name)

            else:
                for i, im in enumerate(data):
                    axes[row][i].imshow(im[channel], cmap=shader)
                    axes[row][i].set_title(name + str(channel))


        else:
            if name == 'raw':
                plt.imshow(data, cmap='gray')
            if name == 'labels':
                plt.imshow(data, alpha=0.5)

    row = 0
    if raw is not None:
        wrapper(raw, row=row)
        row += 1
    if ground_truth is not None:
        wrapper(ground_truth, row=row, name='labels')
        row += 1
    if target is not None:
        wrapper(target, row=row, name=target_name)
        row += 1
    if prediction is not None:
        wrapper(prediction, row=row, name=prediction_name)
        row += 1
    # for label in axes.xaxis.get_tick_labels()[1::2]:
    #    print(len(label.get_text()))
    tb.add_figure(axes[0][0].title.get_text(), fig, it)
    return plt
    # plt.show()

