# Based on https://plotly.com/python/visualizing-mri-volume-slices/ by Emilia Petrisor
# Changes:
# - Don't hardcode shapes
# - Adapt to neuron data
# - Add segmentation mask overlays

# TODO: Downsample (esp. z axis)


import time
import numpy as np

from skimage import io

# vol = io.imread("https://s3.amazonaws.com/assets.datacamp.com/blog_assets/attention-mri.tif")

from nseg.sandbox.videowandb import get_data


z_subsample = 8

raw, lab = get_data(dchw_rgb=False)
vol = raw
lab = lab.astype(np.uint8)

if z_subsample != 1:
    vol = vol[::z_subsample]
    lab = lab[::z_subsample]

volume = vol#.T

nz, ny, nx = volume.shape
# TODO: Are ny and nx in the correct order?


# Define frames
import plotly.graph_objects as go
import plotly

nb_frames = nz
z_init = nz - 1

# fig = go.Figure(
#     frames=[go.Frame(
#         data=go.Surface(
#             z=(nz * 0.1 - k * 0.1) * np.ones((ny, nx)),
#             surfacecolor=np.flipud(volume[z_init - k]),
#             cmin=0, cmax=255
#         ),
#         name=str(k) # you need to name the frame for the animation to behave properly
#     )
#     for k in range(nb_frames)]
# )

fig = go.Figure(
    frames=[go.Frame(
        data=[
            go.Surface(
                z=(nz * 0.1 - k * 0.1) * np.ones((ny, nx)),
                surfacecolor=np.flipud(volume[z_init - k]),
                colorscale='Gray',
                opacity=0.9,
                cmin=0, cmax=255
            ),
            go.Surface(
                z=(nz * 0.1 - k * 0.1) * np.ones((ny, nx)),
                surfacecolor=np.flipud(lab[z_init - k]),
                colorscale=plotly.colors.qualitative.Alphabet,
                opacity=0.2,
                # colorbar=dict(thickness=20, ticklen=4),
            ),
        ],
        name=str(k) # you need to name the frame for the animation to behave properly
    )
    for k in range(nb_frames)]
)

# fig = go.Figure()

# Add data to be displayed before animation starts
fig.add_trace(go.Surface(
    z=nz * 0.1 * np.ones((ny, nx)),
    surfacecolor=np.flipud(volume[z_init]),
    colorscale='Gray',
    cmin=0, cmax=255,
    opacity=0.9,
    # colorbar=dict(thickness=20, ticklen=4),
))

# Labels

fig.add_trace(go.Surface(
    z=nz * 0.1 * np.ones((ny, nx)),
    surfacecolor=np.flipud(lab[z_init]),
    colorscale=plotly.colors.qualitative.Alphabet,
    opacity=0.2,
    # colorbar=dict(thickness=20, ticklen=4),
))


# fig.show()
# exit()

def frame_args(duration):
    return {
        "frame": {"duration": duration},
        "mode": "immediate",
        "fromcurrent": True,
        "transition": {"duration": duration, "easing": "linear"},
    }

sliders = [{
    "pad": {"b": 10, "t": 60},
    "len": 0.9,
    "x": 0.1,
    "y": 0,
    "steps": [
        {
            "args": [[f.name], frame_args(0)],
            "label": str(k),
            "method": "animate",
        }
        for k, f in enumerate(fig.frames)
    ],
}]

# Layout
fig.update_layout(
    title='Slices in volumetric data',
    width=600,
    height=600,
    scene=dict(
        zaxis=dict(range=[-0.1, (nz + 1) * 0.1], autorange=False),
        aspectratio=dict(x=1, y=1, z=1),
    ),
    updatemenus = [
    {
        "buttons": [
            {
                "args": [None, frame_args(50)],
                "label": "&#9654;", # play symbol
                "method": "animate",
            },
            {
                "args": [[None], frame_args(0)],
                "label": "&#9724;", # pause symbol
                "method": "animate",
            },
        ],
        "direction": "left",
        "pad": {"r": 10, "t": 70},
        "type": "buttons",
        "x": 0.1,
        "y": 0,
    }
    ],
    sliders=sliders
)

fig.show()

# exit()

import wandb

wandb.init(
    project="ptest",
    notes="plotly logging experiment",
)
# wandb.log({'plotlyfig': fig})  # Doesn't animate -> workaround: use html export/import
wandb.log({'plotlyfig': wandb.Html(plotly.io.to_html(fig))})
