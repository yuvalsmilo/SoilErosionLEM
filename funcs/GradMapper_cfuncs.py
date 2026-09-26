
import numpy as np
import cfuncs_ErosionDeposition
from funcs.GradMapper import GradMapper


class GradMapper_cfuncs(GradMapper):
    _name = "GradMapper"
    _unit_agnostic = True
    _info = {
        'topographic__elevation': {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Land surface elevation",
        },
        'surface_water__depth': {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Depth of water on the land surface",
        },
        'water_surface__elevation': {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": ("Elevation of the water surface (topographic__elevation "
                    "+ surface_water__depth)")
        },
        'water_surface__slope': {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m/m",
            "mapping": "node",
            "doc": ("Steepest outward (downwind) water-surface gradient "
                    "magnitude at each node."),
        },
        'topographic__gradient': {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m/m",
            "mapping": "link",
            "doc": "Topographic gradient at link",
        },
        'downwind__link_gradient': {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m/m",
            "mapping": "node",
            "doc": "Gradient of link to downwind node at node",
        },
        'upwind__link_gradient': {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m/m",
            "mapping": "node",
            "doc": "Gradient of link to upwind node at node",
        },
    }

    def __init__(self, grid, minslope=0.001):
        super().__init__(grid, minslope=minslope)

        self._row_max_buffer = np.zeros(self._grid.number_of_nodes)
        self._row_min_buffer = np.zeros(self._grid.number_of_nodes)

    def run_one_step(self, ):
        # positive link direction is INCOMING
        gradient_of_downwind_link_at_node = self._grid.at_node['downwind__link_gradient']
        gradient_of_upwind_link_at_node = self._grid.at_node['upwind__link_gradient']
        topographic_gradient_at_link = self._grid.at_link['topographic__gradient']
        gradients_vals = self._grid.at_node['water_surface__slope']
        topographic_gradient_at_link[:] = self._grid.calc_grad_at_link('topographic__elevation')

        links_at_node = self._grid.links_at_node
        link_dirs_at_node = self._grid.link_dirs_at_node
        shape = (self._grid.number_of_nodes,)

        # Map the largest magnitude of the links bringing flux into the node
        # to the node.
        values_at_links = topographic_gradient_at_link[links_at_node] * link_dirs_at_node
        _, row_min = cfuncs_ErosionDeposition.calc_row_max_min_4(
            values_at_links, self._row_max_buffer, self._row_min_buffer, shape)
        gradient_of_upwind_link_at_node[:] = row_min
        gradient_of_upwind_link_at_node[gradient_of_upwind_link_at_node > 0] = 0  # POSITIVE ARE OUTFLUX
        gradient_of_upwind_link_at_node[:] = np.abs(gradient_of_upwind_link_at_node)

        topographic_gradient_at_link[self._inactive_links] = 0
        values_at_links = topographic_gradient_at_link[links_at_node] * link_dirs_at_node
        steepest_links_at_node, _ = cfuncs_ErosionDeposition.calc_row_max_min_4(
            values_at_links, self._row_max_buffer, self._row_min_buffer, shape)

        gradient_of_downwind_link_at_node[:] = 0  # set all to zero
        # if maximal link is  negative, it will be zero. meaning, no outflux
        gradient_of_downwind_link_at_node[:] = np.fmax(steepest_links_at_node,
                                                       gradient_of_downwind_link_at_node)
        gradient_of_downwind_link_at_node[gradient_of_downwind_link_at_node <= self._minslope] = 0

        self._grid.at_node['water_surface__elevation'][self._grid.core_nodes] = \
        self._grid.at_node['surface_water__depth'][self._grid.core_nodes] + \
        self._grid.at_node['topographic__elevation'][self._grid.core_nodes]

        water_gradient_at_link = self._grid.calc_grad_at_link('water_surface__elevation')
        gradient_to_outlet = topographic_gradient_at_link[
            self._outlet_links]  # ! ACCORDING TO THE TOPOGRAPHIC SLOPE AND NOT THE WATER SLOPE.
        gradient_to_outlet[np.abs(gradient_to_outlet) <= self._minslope] = 0
        water_gradient_at_link[self._outlet_links] = gradient_to_outlet
        water_gradient_at_link[self._inactive_links] = 0

        values_at_links = water_gradient_at_link[links_at_node] * link_dirs_at_node
        steepest_links_at_node, _ = cfuncs_ErosionDeposition.calc_row_max_min_4(
            values_at_links, self._row_max_buffer, self._row_min_buffer, shape)
        gradients_vals[:] = 0
        watergradient_of_downwind_link_at_node = np.fmax(steepest_links_at_node,
                                                         gradients_vals)  # if maximal link is negative, it will be
        # equal zero == no outflux
        watergradient_of_downwind_link_at_node[watergradient_of_downwind_link_at_node <= self._minslope] = 0
        gradients_vals[self.grid.core_nodes] = watergradient_of_downwind_link_at_node[self.grid.core_nodes]
