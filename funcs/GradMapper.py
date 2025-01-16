import numpy as np
from landlab import Component
import time


class GradMapper(Component):
    _name = "GradMapper"
    _unit_agnostic = True
    _info = {
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

    def __init__(
            self,
            grid,
            minslope=0.001,
    ):
        super(GradMapper, self).__init__(grid)

        grid.add_zeros('topographic__gradient', at="link")
        grid.add_zeros('downwind__link_gradient', at="node")
        grid.add_zeros('upwind__link_gradient', at="node")
        self._minslope = minslope
        self._inactive_links = self._grid.status_at_link == self._grid.BC_LINK_IS_INACTIVE
        self._outlet_links = self._grid.links_at_node[self.grid.nodes.flatten()[self.grid._node_status == 1]]

    def run_one_step(self, ):
        # positive link direction is INCOMING
        gradient_of_downwind_link_at_node = self._grid.at_node['downwind__link_gradient']
        gradient_of_upwind_link_at_node = self._grid.at_node['upwind__link_gradient']
        topographic_gradient_at_link = self._grid.at_link['topographic__gradient']
        gradients_vals = self._grid.at_node['water_surface__slope']
        topographic_gradient_at_link[:] = self._grid.calc_grad_at_link('topographic__elevation')

        # Map the largest magnitude of the links bringing flux into the node to the node
        gradient_of_upwind_link_at_node[:] = -self._grid.map_upwind_node_link_max_to_node(
            topographic_gradient_at_link)  
        gradient_of_upwind_link_at_node[gradient_of_upwind_link_at_node > 0] = 0  # POSITIVE ARE OUTFLUX
        gradient_of_upwind_link_at_node[:] = np.abs(gradient_of_upwind_link_at_node)

        topographic_gradient_at_link[self._inactive_links] = 0
        values_at_links = (topographic_gradient_at_link[
                              self._grid.links_at_node] *
                           self._grid.link_dirs_at_node)  # this procedure makes incoming links NEGATIVE
        steepest_links_at_node = np.amax(values_at_links, axis=1)  # take the maximum (positive means out link)

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

        values_at_links = (water_gradient_at_link[
                              self._grid.links_at_node] *
                           self._grid.link_dirs_at_node)  # this procedure makes incoming links NEGATIVE
        steepest_links_at_node = np.amax(values_at_links, axis=1)  # take the maximum (positive means out link)
        gradients_vals[:] = 0
        watergradient_of_downwind_link_at_node = np.fmax(steepest_links_at_node,
                                                         gradients_vals)  # if maximal link is negative, it will be 
        # equal zero == no outflux
        watergradient_of_downwind_link_at_node[watergradient_of_downwind_link_at_node <= self._minslope] = 0
        gradients_vals[self.grid.core_nodes] = watergradient_of_downwind_link_at_node[self.grid.core_nodes]

    def flux_mapper(self):
        "get sediment load fluxes [kg/s]"
        total_sediment_incoming_flux = np.copy(self._grid.at_node['sediment__influx'])

        return total_sediment_incoming_flux
