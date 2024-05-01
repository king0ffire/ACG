# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import matplotlib.pyplot as plt


def image_grid(
    images,
    rows=None,
    cols=None,
    fill: bool = True,
    show_axes: bool = False,
    rgb: bool = True,
):
    """
    A util function for plotting a grid of images.

    Args:
        images: (N, H, W, 4) array of RGBA images
        rows: number of rows in the grid
        cols: number of columns in the grid
        fill: boolean indicating if the space between images should be filled
        show_axes: boolean indicating if the axes of the plots should be visible
        rgb: boolean, If True, only RGB channels are plotted.
            If False, only the alpha channel is plotted.

    Returns:
        None
    """
    if (rows is None) != (cols is None):
        raise ValueError("Specify either both rows and cols or neither.")

    if rows is None:
        rows = len(images)
        cols = 1

    gridspec_kw = {"wspace": 0.0, "hspace": 0.0} if fill else {}
    fig, axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw, figsize=(15, 9))
    bleed = 0
    fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

    for ax, im in zip(axarr.ravel(), images):
        if rgb:
            # only render RGB channels
            ax.imshow(im[..., :3])
        else:
            # only render Alpha channel
            ax.imshow(im[..., 3])
        if not show_axes:
            ax.set_axis_off()
            
            

# Copied from PyTorch3D repository to fix line #36 while it's not fixed there
# fig.gca(projection="3d") -> fig.add_subplot(projection="3d")
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from pytorch3d.vis.plotly_vis import get_camera_wireframe


def plot_cameras(ax, cameras, color: str = "blue"):
    """
    Plots a set of `cameras` objects into the maplotlib axis `ax` with
    color `color`.
    """
    cam_wires_canonical = get_camera_wireframe().cuda()[None]
    cam_trans = cameras.get_world_to_view_transform().inverse()
    cam_wires_trans = cam_trans.transform_points(cam_wires_canonical)
    plot_handles = []
    for wire in cam_wires_trans:
        # the Z and Y axes are flipped intentionally here!
        x_, z_, y_ = wire.detach().cpu().numpy().T.astype(float)
        (h,) = ax.plot(x_, y_, z_, color=color, linewidth=0.3)
        plot_handles.append(h)
    return plot_handles


def plot_camera_scene(cameras, cameras_gt, status: str):
    """
    Plots a set of predicted cameras `cameras` and their corresponding
    ground truth locations `cameras_gt`. The plot is named with
    a string passed inside the `status` argument.
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.clear()
    ax.set_title(status)
    handle_cam = plot_cameras(ax, cameras, color="#FF7D1E")
    handle_cam_gt = plot_cameras(ax, cameras_gt, color="#812CE5")
    plot_radius = 3
    ax.set_xlim3d([-plot_radius, plot_radius])
    ax.set_ylim3d([3 - plot_radius, 3 + plot_radius])
    ax.set_zlim3d([-plot_radius, plot_radius])
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")
    labels_handles = {
        "Estimated cameras": handle_cam[0],
        "GT cameras": handle_cam_gt[0],
    }
    ax.legend(
        labels_handles.values(),
        labels_handles.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
    )
    plt.show()
    return fig