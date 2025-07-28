#!/usr/bin/env python

import toyplot.svg
import toytree
import xml.etree.ElementTree as ElementTree
import ipywidgets as widgets
import k3d
import numpy as np
from IPython.display import display


def visualize_3d_over_tree(
        tree,  # toytree generated tree object
        object_list,  # list of k3d.mesh objects or None, length must match tree node count
        canvas_shape,  # (c_width, c_height) tuple for canvas dimensions
        object_shape,  # (o_width, o_height) tuple for mini-plot dimensions
        tree_draw_params=None,  # additional parameters for tree.draw (excluding width/height)
        obj_camera=None,  # k3d camera configuration, None for auto-fit
        obj_x_offset=0.,  # TODO: add a toggle to do this
        obj_y_offset=0.,  # TODO: add a toggle to do this
):
    """
    Creates a combined visualization of a phylogenetic tree with 3D objects at each node.
    # TODO: add a function to rotate around z-axis
    # TODO: add a widget to mimic k3d panel but control all objects

    Parameters:
        tree: toytree object - The phylogenetic tree to visualize
        object_list: list - List of k3d objects (e.g., k3d.mesh) or None.
                     Must have the same length as the number of nodes in the tree.
                     None entries skip creating a plot for that node.
        canvas_shape: tuple - (width, height) of the main canvas in pixels
        object_shape: tuple - (width, height) of each mini 3D plot in pixels
        tree_draw_params: dict - Additional parameters to pass to tree.draw()
        obj_camera: list - k3d camera configuration [x, y, z, target_x, target_y, target_z, up_x, up_y, up_z]
                     None enables auto-fit camera
        obj_x_offset: float - Offset for positioning plots in x-direction
        obj_y_offset: float - Offset for positioning plots in y-direction

    Returns:
        ipywidgets.Layout - The complete visualization layout
    """
    # validate input parameters
    num_nodes = tree.nnodes
    assert len(object_list) == num_nodes, \
        f"object_list length ({len(object_list)}) must match number of tree nodes ({num_nodes})"

    c_width, c_height = canvas_shape
    o_width, o_height = object_shape
    tree_draw_params = tree_draw_params or {}

    # generate tree and draw with provided parameters
    canvas, axes, mark = tree.draw(
        width=c_width,
        height=c_height,
        **tree_draw_params
    )
    svg_data = toyplot.svg.render(canvas)
    svg_str = ElementTree.tostring(svg_data, method="xml").decode("utf8")

    # base container
    container = widgets.Box(
        [],
        layout=widgets.Layout(
            width=f'{c_width}px',
            height=f'{c_height}px',
            position='relative',
            overflow='visible',
            margin='0',
            padding='0',
            border='none'
        )
    )

    # tree SVG widget
    tree_widget = widgets.HTML(
        f'<div style="width:{c_width}px; height:{c_height}px; position:absolute;">{svg_str}</div>'
    )

    # get node coordinates in pixel space
    ntable = mark.ntable
    node_coordinates = {node.idx: (ntable[node.idx][0], ntable[node.idx][1]) for node in tree}

    def convert_to_pixel(x, y):
        px = axes.project("x", x)
        py = axes.project("y", y)
        return px, py

    node_coordinates_pixel = {
        node: convert_to_pixel(x, y) for node, (x, y) in node_coordinates.items()
    }

    # build k3d mini-plots for nodes with valid objects
    plots = []
    k3d_widgets = []
    # offset for positioning plots due to that every time a widget was created,
    # there will be a same-width offset
    accumulated_x_offset = 0

    for i, (node, (x, y)) in enumerate(node_coordinates_pixel.items()):
        # skip nodes with None in object_list
        k3d_obj = object_list[i]
        if k3d_obj is None:
            continue

        # create plot for this node
        plot = k3d.plot()
        plot.width = o_width
        plot.height = o_height
        plot.menu_visibility = False
        plot.axes_helper = 0
        plot.grid_visible = False

        # configure camera based on input parameter
        if obj_camera is None:
            plot.camera_auto_fit = True
        else:
            plot.camera_auto_fit = False
            plot.camera = obj_camera

        # add the 3D object to the plot
        plot += k3d_obj

        # position the plot absolutely
        plot.layout = widgets.Layout(
            position='absolute',
            left=f'{x - o_width / 2 - accumulated_x_offset + obj_x_offset}px',
            top=f'{y - o_height / 2 + obj_y_offset}px',
            width=f'{o_width}px',
            height=f'{o_height}px',
            overflow='visible',
            border='none',
            margin='0',
            padding='0'
        )

        # update tracking lists and offset
        plots.append(plot)
        k3d_widgets.append(plot)
        accumulated_x_offset += o_width  # Only increment for valid plots

    # control toggles
    sync_toggle = widgets.ToggleButton(
        value=True,
        description="Sync All",
        tooltip="Toggle camera & param sync"
    )

    # sync logic
    sync_enabled = {"value": sync_toggle.value}
    updating = {"active": False}

    def sync_all(source_plot):
        for p_ in plots:
            if p_ is not source_plot:
                p_.camera = list(source_plot.camera)
                p_.lighting = source_plot.lighting
                for src_obj, tgt_obj in zip(source_plot.objects, p.objects):
                    for prop_ in ["opacity", "point_size", "shader", "visible"]:
                        if hasattr(src_obj, prop_) and hasattr(tgt_obj, prop_):
                            setattr(tgt_obj, prop_, getattr(src_obj, prop_))

    def make_observer(plot_):
        def callback(change):
            if sync_enabled["value"] and not updating["active"]:
                updating["active"] = True
                sync_all(plot_)
                updating["active"] = False

        return callback

    # Attach observers to plots and their objects
    for p in plots:
        p.observe(make_observer(p), names="camera")
        p.observe(make_observer(p), names="lighting")
        for obj in p.objects:
            for prop in ["opacity", "point_size", "shader", "visible"]:
                obj.observe(make_observer(p), names=prop)

    # Sync toggle behavior
    sync_toggle.observe(lambda ch: sync_enabled.update(value=ch["new"]), names='value')

    # Assemble final layout
    container.children = (tree_widget,) + tuple(k3d_widgets)
    visualization_layout = widgets.VBox([
        widgets.HBox([sync_toggle]),
        widgets.Box(
            [container],
            layout=widgets.Layout(
                position='relative',
                width=f'{c_width}px',
                height=f'{c_height}px',
                overflow='visible',
                display='block'
            )
        )
    ])

    return visualization_layout


# Example usage:
def test_visualize_3d_over_tree():
    # Create a sample tree
    tree = toytree.rtree.unittree(ntips=5)

    # Define tetrahedron as sample object
    tetra_vertices = np.array([
        [0, 0, 15],
        [0, 15, -5],
        [-13, -7, -5],
        [13, -7, -5]
    ], dtype=np.float32)

    tetra_indices = np.array([
        0, 1, 2,
        0, 2, 3,
        0, 3, 1,
        1, 3, 2
    ], dtype=np.uint32)

    # Create object list (one per node, using different colors)
    object_list = []
    for i in range(tree.nnodes):
        color = 0x0000ff + (i * 0x202020) % 0xffffff
        obj = k3d.mesh(tetra_vertices, tetra_indices, color=color)
        object_list.append(obj)

    # Create visualization
    viz_layout = visualize_3d_over_tree(
        tree=tree,
        object_list=object_list,
        canvas_shape=(500, 400),
        object_shape=(40, 40),
        # tree_draw_params={"tip_labels": False},
        # tree_draw_params={"tip_labels_align": True},
        obj_camera=[0, 0, 30, 0, 0, 0, 0, 1, 0],
        obj_x_offset=-30,
        obj_y_offset=-20
    )

    # Display the visualization
    display(viz_layout)


if __name__ == "__main__":
    test_visualize_3d_over_tree()
