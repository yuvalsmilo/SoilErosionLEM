"""
"""

import numpy as np
import scipy.constants
from landlab import Component
from cfuncs_erosion_deposition import cfuncs_ErosionDeposition

class OverlandflowErosionDeposition(Component):
    """
    """

    _name = "OverlandflowErosionDeposition"

    _unit_agnostic = True

    _info = {
        "surface_water__depth": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Depth of water on the surface",
        },
        "topographic__elevation": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Land surface topographic elevation",
        },
        "topographic__slope": {
            "dtype": float,
            "intent": "in",
            "optional": True,
            "units": "-",
            "mapping": "node",
            "doc": "gradient of the ground surface",
        },
        "sediment_load__weight": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "kg",
            "mapping": "node",
            "doc": "weight of sediment load at node",
        },
        "sediment__influx": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m3/s",
            "mapping": "node",
            "doc": "Sediment flux (volume per unit time of sediment entering each node)",
        },
        "sediment__outflux": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m3/s",
            "mapping": "node",
            "doc": "Sediment flux (volume per unit time of sediment leaving each node)",
        },
    }

    def __init__(
            self,
            grid,
            fluid_density=1000.0,
            g=scipy.constants.g,
            sigma=2000,
            tau_crit=1.165,
            cover_depth_star=0.1,
            phi=0.4,
            slope="topographic__slope",
            bedrock_sediment_grainsizes=None,
            R=1.65,
            C1=18,
            C2=0.4,
            v=9.2 * 10 ** -7,
            alpha=0.045,  # 0.045 from Komar 1987
            beta=-0.68,  # -0.68 From Komar 1987
            TC_model='YALIN',
            change_topo_flag=True,
            kr=0.0002,
            max_flipped_deposition_slope=0.01,  # [m/m] -- allow inverse of topography up to certain slope
            depression_depth=0.0055 # Charteristic surface depression depth/height [m]. Used to calculate flow 'width' in each cell

    ):

        super().__init__(grid)

        assert slope in grid.at_node
        self.initialize_output_fields()
        self._zeros_at_link = self._grid.zeros(at="link")
        self._zeros_at_node = self._grid.zeros(at="node")

        self._slope = slope
        self._g = g
        self._rho = fluid_density
        self._sigma = sigma
        self._phi = phi
        self._cover_depth_star = cover_depth_star

        self._R = R
        self._C1 = C1
        self._C2 = C2
        self._v = v
        self._SG = self._sigma / self._rho
        self._alpha = alpha
        self._beta = beta
        self._TC_model = TC_model
        self._kr = kr

        self._bedrock_sediment_grainsizes = bedrock_sediment_grainsizes
        self._nodes_flatten = grid.nodes.flatten().astype('int')
        self._nodes = np.shape(self._grid.nodes)[0] * np.shape(self._grid.nodes)[1]

        self._inactive_links = grid.status_at_link == grid.BC_LINK_IS_INACTIVE
        self._active_links = ~(grid.status_at_link == grid.BC_LINK_IS_INACTIVE)
        self._links_array  = np.arange(0,np.size(self._zeros_at_link)).tolist()
        self._active_links_ids = np.array(self._links_array)[self._active_links]

        self._n_grain_sizes = np.shape(self._grid.at_node['grains__weight'])[1]
        self._zeros_at_node_for_fractions = np.zeros((self._nodes, np.shape(self._grid.at_node['grains__weight'])[1]))
        self._zeros_at_links_for_fractions = np.zeros(
            (np.size(self._zeros_at_link), np.shape(self._grid.at_node['grains__weight'])[1]))
        self._sediment_load_flux_dzdt_at_node_per_size = self._zeros_at_node_for_fractions

        ## Variables with dimensions of nodes/links x grain_sizes
        self._weight_flux_at_link = np.zeros(
            (np.size(self._zeros_at_link), np.shape(self._grid.at_node['grains__weight'])[1]))
        self._sediment_load_concentration_at_link = np.zeros(
            (np.size(self._zeros_at_link), np.shape(self._grid.at_node['grains__weight'])[1]))
        self._DR = np.zeros((self._nodes, np.shape(self._grid.at_node['grains__weight'])[1]))
        self._tau_crit = np.zeros((self._nodes, np.shape(self._grid.at_node['grains__weight'])[1]))
        self._tau_crit[:] = tau_crit
        self._outlinks_fluxes_at_node = np.zeros((self._nodes, 4, np.shape(self._grid.at_node['grains__weight'])[1]))
        self._inlinks_fluxes_at_node = np.zeros((self._nodes, 4, np.shape(self._grid.at_node['grains__weight'])[1]))
        self._c_si = np.zeros((self._nodes, np.shape(self._grid.at_node['grains__weight'])[1]))
        self._c_kg = np.zeros((self._nodes, np.shape(self._grid.at_node['grains__weight'])[1]))
        self._grain_fractions_at_node = np.zeros((self._nodes, np.shape(self._grid.at_node['grains__weight'])[1]))
        self._suspended_fraction_at_node = np.zeros((self._nodes, np.shape(self._grid.at_node['grains__weight'])[1]))
        self._TC = np.zeros((self._nodes, np.shape(self._grid.at_node['grains__weight'])[1]))

        self._change_topo_flag = change_topo_flag
        self._xy_spacing = np.array(self.grid.spacing)
        self._grid_shape = np.array(self.grid.shape)
        self._min_sediment_load_weight_to_del = 10 ** -10
        self._max_flipped_deposition_slope = max_flipped_deposition_slope
        self._depression_depth = depression_depth
        self._bedrock_grain_fractions = self._grid.at_node["bed_grains__proportions"][self._grid.core_nodes[0]]
        self._bedrock_median_size = self._grid.at_node["grains_classes__size"][self._grid.core_nodes[0]][int(np.argwhere(np.cumsum(self._bedrock_grain_fractions) >= 0.5)[0])]
        if self._bedrock_sediment_grainsizes == None:
            self._bedrock_sediment_grainsizes = self._grid.at_node["grains_classes__size"][self._grid.core_nodes[0]]

        self._sediments_flux_at_link = np.zeros(
            (np.shape(self._zeros_at_link)[0], np.shape(self._grid.at_node['grains__weight'])[1]))
        self._sediment_load_weight_at_node_per_size = np.zeros(
            (np.shape(self._zeros_at_node)[0], np.shape(self._grid.at_node['grains__weight'])[1]))
        self._deposited_sediments_dz_at_node = self._grid.zeros(at="node")
        self._detached_bedrock_rate_dz = np.zeros_like(self._zeros_at_node)
        self._detached_soil_rate_dz = np.zeros_like(self._zeros_at_node)
        self._detached_bedrock_weight = np.zeros(
            (np.shape(self._zeros_at_node)[0], np.shape(self._grid.at_node['grains__weight'])[1]))
        self._detached_soil_weight = np.zeros(
            (np.shape(self._zeros_at_node)[0], np.shape(self._grid.at_node['grains__weight'])[1]))
        self._deposited_sediments_weights_at_node = np.zeros(
            (np.shape(self._zeros_at_node)[0], np.shape(self._grid.at_node['grains__weight'])[1]))
        self._vs = np.zeros((1, np.shape(self._grid.at_node['grains__weight'])[
            1]))  # ratio of near-bed sediment concentration to the vertically averaged concentration
        
        
        self._calc_settling_velocity_per_size()
    
    
    def _calc_settling_velocity_per_size(self):
        # Based on: https://pubs.geoscienceworld.org/sepm/jsedres/article/74/6/933/99413/A-Simple-Universal-Equation-for-Grain-Settling
        for i, g_size in enumerate(self._grid.at_node["grains_classes__size"][self._grid.core_nodes[0]]):
            self._vs[0, i] = np.divide(
                (self._R * self._g * g_size ** 2),
                (self._C1 * self._v + (0.75 * self._C2 * self._R * self._g * (g_size) ** 3) ** 0.5)
            )  
        

    def _init_variables(self):


        self._detached_soil_weight.fill(0.0)
        self._detached_bedrock_weight.fill(0.0)
        self._detached_bedrock_rate_dz.fill(0.0)
        self._detached_soil_rate_dz.fill(0.0)
        self._sediment_load_flux_dzdt_at_node_per_size.fill(0.0)
        self._deposited_sediments_weights_at_node.fill(0.0)
        self._deposited_sediments_dz_at_node.fill(0.0)
        self._outlinks_fluxes_at_node.fill(0.0)
        self._inlinks_fluxes_at_node.fill(0.0)
        self._sediment_load_concentration_at_link.fill(0.0)
        self._grid.at_node['sediment__influx'].fill(0.0)
        self._grid.at_node['sediment__outflux'].fill(0.0)
        self._sediments_flux_at_link.fill(0.0)
        self._c_si.fill(1.0)
        self._c_kg.fill(1.0)
        self._grain_fractions_at_node.fill(0.0)
        self._suspended_fraction_at_node.fill(0.0)
        self._weight_flux_at_link.fill(0.0)



    def _calc_inout_fluxes_at_node(self,
                     upwind_node_ids_at_link,
                     downwind_node_ids_at_link,
                     weight_flux_at_link,
                     sediment_load_weight_at_node_per_size,
                     outlinks_at_node):

        outlinks_fluxes_at_node = np.zeros_like(self._zeros_at_node_for_fractions)
        inlinks_fluxes_at_node = np.zeros_like(self._zeros_at_node_for_fractions)
        shape = [np.size(self._active_links_ids), self._n_grain_sizes]
        total_outflux_at_node = np.zeros_like(self._zeros_at_node)
        total_influx_at_node = np.zeros_like(self._zeros_at_node)

        (outlinks_fluxes_at_node,
         inlinks_fluxes_at_node,
         total_outflux_at_node,
         total_influx_at_node) = cfuncs_ErosionDeposition.get_outin_fluxes(
            upwind_node_ids_at_link,
            downwind_node_ids_at_link,
            weight_flux_at_link,
            self._active_links_ids,
            outlinks_fluxes_at_node,
            inlinks_fluxes_at_node,
            total_outflux_at_node,
            total_influx_at_node,
            shape)

        indices_to_correct_flux = np.where(outlinks_fluxes_at_node > sediment_load_weight_at_node_per_size)
        if np.any(indices_to_correct_flux):
            ratios = np.divide(sediment_load_weight_at_node_per_size[indices_to_correct_flux],
                               outlinks_fluxes_at_node[indices_to_correct_flux])

            for i, (n, gs) in enumerate(zip(indices_to_correct_flux[0], indices_to_correct_flux[1])):
                out_links = self._grid.links_at_node[n, :][outlinks_at_node[n, :]]
                weight_flux_at_link[out_links, gs] *= ratios[i, np.newaxis] # Update the weight flux at link.

            outlinks_fluxes_at_node = np.zeros_like(self._zeros_at_node_for_fractions)
            inlinks_fluxes_at_node = np.zeros_like(self._zeros_at_node_for_fractions)
            total_outflux_at_node = np.zeros_like(self._zeros_at_node)
            total_influx_at_node = np.zeros_like(self._zeros_at_node)

            (outlinks_fluxes_at_node,
             inlinks_fluxes_at_node,
             total_outflux_at_node,
             total_influx_at_node) = cfuncs_ErosionDeposition.get_outin_fluxes(
                upwind_node_ids_at_link,
                downwind_node_ids_at_link,
                weight_flux_at_link,
                self._active_links_ids,
                outlinks_fluxes_at_node,
                inlinks_fluxes_at_node,
                total_outflux_at_node,
                total_influx_at_node,
                shape)

        self._grid.at_node['sediment__influx'][:] = total_influx_at_node
        self._grid.at_node['sediment__outflux'][:] = total_outflux_at_node

        return weight_flux_at_link, outlinks_fluxes_at_node, inlinks_fluxes_at_node


    def _calc_dzdt(self, size_class, dt=1):

        size_class = int(size_class)
        dt = dt
        xy_spacing = self._xy_spacing
        shape = self._grid_shape
        out = np.zeros_like(self._zeros_at_node)
        dzdt = cfuncs_ErosionDeposition.calc_flux_div_at_node(shape,
                                                       xy_spacing,
                                                       self._sediments_flux_at_link[:, size_class] * dt,
                                                       out)

        dzdt = -dzdt
        return dzdt

    def _calc_sediments_flux_at_link(self):

        ## Calc net suspended sediment flux
        size_class = np.where(np.any(self._sediments_flux_at_link, axis=0))[0].tolist()
        if np.any(size_class):
            if np.size(size_class) > 1:
                result = map(self._calc_dzdt,
                             size_class,
                             np.ones_like(size_class))
                self._sediment_load_flux_dzdt_at_node_per_size[:, size_class] = np.asarray(list(result)).T

            else:
                dzdt = self._calc_dzdt(size_class=size_class[0])
                self._sediment_load_flux_dzdt_at_node_per_size[:, size_class] = dzdt[:, np.newaxis]


    def _calc_DR(self):

        ## Pointers
        c_si_kg = self._c_kg
        self._roughness_correction = 1
        surface_water__depth_at_node = self.grid.at_node['surface_water__depth']
        S = self._grid.at_node[self._slope]
        median_sizes = np.copy(self._grid.at_node['median_size__weight'])
        median_sizes[median_sizes == 0] = self._bedrock_median_size


        ## Get outflux water dischrage
        shape = (np.size(self._active_links_ids))
        q_at_node = cfuncs_ErosionDeposition.sum_out_discharge(self._upwind_node_ids_at_link,
                                              np.abs(self._grid.at_link['surface_water__discharge']),
                                              self._active_links_ids,
                                              np.zeros_like(self._zeros_at_node),
                                              shape)

        ## Calc shear stress at node.
        self._tau_s = self._rho * self._g * S * surface_water__depth_at_node * self._roughness_correction


        ## Flow width according to the fraction of the cell coevred by water
        # which is approximated by the relationshpip between surface runoff height and maxomum depression capacity (5.5 mm for shurb)
        # (Nunes et al., 2005, CATENA)
        flow_width = np.divide(surface_water__depth_at_node,
                               self._depression_depth)  # depression depth in meters
        flow_width[
            flow_width > 1] = self._grid.dx  # greater than 1 means water depth is over the depression depth so flow width is the grid node width


        ## Calculation of transport capacity at node
        TC = np.zeros_like(self._zeros_at_node_for_fractions)
        sg_c = self._SG
        rho_c = self._rho
        const_sg_g_rho = self._rho * (self._SG - 1) * self._g
        shape = (np.size(self._grid.core_nodes), np.shape(self._zeros_at_node_for_fractions)[1])
        self._TC = cfuncs_ErosionDeposition.calc_TC(
            self._alpha,
            self._beta,
            median_sizes,
            self._grid.at_node['grains_classes__size'],
            self._tau_s,
            TC,
            sg_c,
            rho_c,
            self._grid.core_nodes,
            const_sg_g_rho,
            shape)

        ## Calculation of Dc (detachment rate)
        self._Dc = cfuncs_ErosionDeposition.calc_Dc(self._tau_s,
                                   self._tau_crit,
                                   self._grid.core_nodes,
                                   np.zeros_like(self._zeros_at_node_for_fractions),
                                   self._kr,
                                   shape)

        ## Get the total weight flux
        CQ = cfuncs_ErosionDeposition.calc_CQ(
            c_si_kg,
            np.zeros_like(self._zeros_at_node_for_fractions),
            q_at_node,
            self._grid.core_nodes,
            shape,
            int(self._grid.dx))

        ## Calcuation of net erosion/deposition at node
        self._DR[:] = cfuncs_ErosionDeposition.calc_DR(flow_width,
                                      CQ,
                                      self._TC,
                                      self._Dc,
                                      self._vs[0, :],
                                      self._grid.core_nodes,
                                      q_at_node,
                                      np.zeros_like(CQ),
                                      self._grid.dx,
                                      shape)
    
    def _calc_load_flux(self):
        
        ## Pointers
        water_surface_grad_at_link = self._grid.calc_grad_at_link('water_surface__elevation')
        sediment_load_concentration_at_link = self._sediment_load_concentration_at_link
        weight_flux_at_link = self._weight_flux_at_link
        sediment_load_weight_at_node_per_size = self._sediment_load_weight_at_node_per_size
        upwind_node_ids_at_link = self._upwind_node_ids_at_link
        downwind_node_ids_at_link = self._downwind_node_ids_at_link
        surface_water__depth_at_node = self.grid.at_node['surface_water__depth']
        surface_water__depth_at_node[surface_water__depth_at_node < 10 ** -8] = 10 ** -8
        surface_water__depth_at_node_expand = np.expand_dims(surface_water__depth_at_node, -1)
        q_water_at_link = self.grid.at_link[
            'surface_water__discharge']  # surface water discharge units are m**3/s -> From OverlandFlow component
        q_water_at_link[self._inactive_links] = 0

        ## Weight concentration at link
        sediment_load_concentration_at_link[:] = np.divide(sediment_load_weight_at_node_per_size[
                                                                  upwind_node_ids_at_link, :],
                                                                  surface_water__depth_at_node_expand[
                                                                  upwind_node_ids_at_link,
                                                                  :] * self._grid.dx * self._grid.dx)
        ## Find the outlinks for each node.
        outlinks_at_node = self.grid.link_at_node_is_downwind(water_surface_grad_at_link)
        
        
        ## Calc weight flux at link
        shape = [np.size(self._active_links_ids), self._n_grain_sizes]
        weight_flux_at_link[:] = cfuncs_ErosionDeposition.calc_flux_at_link_per_size(q_water_at_link,
                                                                                     sediment_load_concentration_at_link,
                                                                                     self._grid.active_links,
                                                                                     np.zeros_like(weight_flux_at_link),
                                                                                     shape)
        weight_flux_at_link = np.abs(weight_flux_at_link)
        
        
        ## Get the in/out suspended weight fluxe at NODE
        weight_flux_at_link[:], outlinks_fluxes_at_node, inlinks_fluxes_at_node = self._calc_inout_fluxes_at_node(
            upwind_node_ids_at_link,
            downwind_node_ids_at_link,
            weight_flux_at_link,
            sediment_load_weight_at_node_per_size,
            outlinks_at_node)

        ## Get the suspended weight flux at LINK
        shape = np.shape(self._sediments_flux_at_link)
        self._sediments_flux_at_link[:] = cfuncs_ErosionDeposition.calc_flux_at_link(self._grid.dx,
                                                                                                self._sigma,
                                                                                                self._phi,
                                                                                                np.abs(
                                                                                                    weight_flux_at_link),
                                                                                                -np.sign(
                                                                                                    water_surface_grad_at_link),
                                                                                                self._sediments_flux_at_link,
                                                                                                shape)

        ## Calculate sediment-load flux at link
        self._calc_sediments_flux_at_link()

    def _map_upwind_downwind_nodes_to_links(self):

        ## Map upwind/downwind node id to links
        upwind_node_ids_at_link = self._grid.map_value_at_max_node_to_link('water_surface__elevation',
                                                                           self._nodes_flatten).astype('int')
        self._upwind_node_ids_at_link = upwind_node_ids_at_link

        downwind_node_ids_at_link = self._grid.map_value_at_min_node_to_link('water_surface__elevation',
                                                                             self._nodes_flatten).astype('int')
        self._downwind_node_ids_at_link = downwind_node_ids_at_link

    def _calc_erosion_deposition(self, ):

        # Pointers
        surface_water__depth_at_node = self.grid.at_node['surface_water__depth']
        surface_water__depth_at_node[surface_water__depth_at_node < 10 ** -8] = 10 ** -8
        surface_water__depth_at_node_expand = np.expand_dims(surface_water__depth_at_node, -1)
        soil_depth = self._grid.at_node['soil__depth']
        grain_weight_at_node = self.grid.at_node['grains__weight']
        q_water_at_link = self.grid.at_link[
            'surface_water__discharge']  # surface water discharge units are m**3/s -> From OverlandFlow component
        q_water_at_link[self._inactive_links] = 0
        deposited_sediment_weights_at_node = self._deposited_sediments_weights_at_node
        sediment_load_weight_at_node_per_size = self._sediment_load_weight_at_node_per_size
        deposited_sediments_dz_at_node = self._deposited_sediments_dz_at_node
        detached_bedrock_rate_dz = self._detached_bedrock_rate_dz  # to suspended
        detached_soil_rate_dz = self._detached_soil_rate_dz  # to suspended
        detached_bedrock_weight = self._detached_bedrock_weight  # to suspended
        detached_soil_weight = self._detached_soil_weight  # to suspended
        
        c_si = self._c_si
        c_kg = self._c_kg
        grain_fractions_at_node = self._grain_fractions_at_node
        suspended_fraction_at_node = self._suspended_fraction_at_node
        grain_weight_at_node[grain_weight_at_node < 10 ** -10] = 10 ** -10


        ## Calc soil exponent
        soil_e_expo = (1 - (np.exp(
            (-soil_depth) / self._cover_depth_star))
                       )

        ## Suspended load total volume
        temp_sediment_load_weight_at_node_per_size = np.copy(sediment_load_weight_at_node_per_size)
        temp_sediment_load_weight_at_node_per_size[temp_sediment_load_weight_at_node_per_size < 0] = 0
        temp_suspended_sediment_volume_at_node_per_size = temp_sediment_load_weight_at_node_per_size / self._sigma  # remmeber, here, no porosity correction
        temp_suspended_sediment_volume_at_node_per_size[temp_suspended_sediment_volume_at_node_per_size < 0] = 0

        ## Concentrations (volumetric and weight)
        c_si[:] = np.divide(temp_suspended_sediment_volume_at_node_per_size,
                            surface_water__depth_at_node_expand * self.grid.dx ** 2,
                            out=np.ones_like(temp_suspended_sediment_volume_at_node_per_size))

        # Weight concentration
        c_kg[:] = np.divide(temp_sediment_load_weight_at_node_per_size,
                            surface_water__depth_at_node_expand * self.grid.dx ** 2,
                            out=np.ones_like(temp_suspended_sediment_volume_at_node_per_size))


        # Calc weight fraction of all size classes
        temp_sediment_load_weight_at_node_per_size[
            temp_sediment_load_weight_at_node_per_size < 10 ** -10] = 10 ** -10
        suspended_fraction_at_node[:] = cfuncs_ErosionDeposition.calc_concentration(
            np.ones_like(temp_sediment_load_weight_at_node_per_size),
            temp_sediment_load_weight_at_node_per_size,
            np.sum(temp_sediment_load_weight_at_node_per_size, 1),
            np.shape(temp_sediment_load_weight_at_node_per_size)
            )


        # If the concentration is greater than 1, all of the suspended sediments needs to be deposited
        sum_c_si = cfuncs_ErosionDeposition.grain_size_sum_at_node(c_si,
                                                  np.zeros_like(self._zeros_at_node),
                                                  np.shape(c_si))


        ## Calculation of erosion/deposition at node
        self._calc_DR()
        self._DR[sum_c_si >= 1, :] = -np.inf  # DR returns in units of kg/(m^2*s)


        ## Calc net change for both layer
        core_nodes = self._grid.core_nodes
        factor_convert_weight_to_dz_c = self._sigma * (1 - self._phi) * self._grid.dx ** 2
        factor_convert_weight_to_dz_bedrock_c = self._sigma * self._grid.dx ** 2
        shape = (np.size(core_nodes), np.shape(self._zeros_at_node_for_fractions)[1])
        dx_c = self._grid.dx

        ## Calc weight concentration at node
        grain_fractions_at_node[:] = cfuncs_ErosionDeposition.calc_concentration(np.ones_like(grain_weight_at_node),
                                                                grain_weight_at_node,
                                                                np.sum(grain_weight_at_node, axis=1),
                                                                np.shape(grain_weight_at_node)
                                                                )
        ## Calc Erosion/Deposition at node
        (detached_soil_weight[:],
         detached_bedrock_weight[:],
         detached_soil_rate_dz[:],
         detached_bedrock_rate_dz[:],
         deposited_sediment_weights_at_node[:],
         deposited_sediments_dz_at_node[:]) = cfuncs_ErosionDeposition.calc_detached_deposited(self._DR,
                                                                                            np.abs(self._DR),
                                                                                            grain_weight_at_node,
                                                                                            grain_fractions_at_node,
                                                                                            detached_soil_weight,
                                                                                            detached_bedrock_weight,
                                                                                            suspended_fraction_at_node,
                                                                                            self._bedrock_grain_fractions,
                                                                                            temp_sediment_load_weight_at_node_per_size,
                                                                                            deposited_sediment_weights_at_node,
                                                                                            deposited_sediments_dz_at_node,
                                                                                            detached_soil_rate_dz,
                                                                                            detached_bedrock_rate_dz,
                                                                                            soil_e_expo,
                                                                                            core_nodes,
                                                                                            factor_convert_weight_to_dz_c,
                                                                                            factor_convert_weight_to_dz_bedrock_c,
                                                                                            shape,
                                                                                            dx_c
                                                                                            )

    def _calc_stable_dt(self):

        max_downwind_gradient = self._grid.at_node['downwind__link_gradient']
        max_upwind_gradient = self._grid.at_node['upwind__link_gradient']
        S = self._grid.at_node[self._slope]
        detached_bedrock_rate_dz = self._detached_bedrock_rate_dz  # to suspended
        detached_soil_rate_dz = self._detached_soil_rate_dz  # to suspended
        deposited_sediments_dz_at_node = self._deposited_sediments_dz_at_node
        deposited_sediment_weights_at_node = self._deposited_sediments_weights_at_node
        temp_sediment_load_weight_at_node_per_size = np.copy(self._sediment_load_weight_at_node_per_size)
        temp_sediment_load_weight_at_node_per_size[temp_sediment_load_weight_at_node_per_size < 0] = 0
        sediment_load_weight_at_node_per_size = self._sediment_load_weight_at_node_per_size
        outlinks_fluxes_at_node = self._outlinks_fluxes_at_node



        # Condition 1:
        # Stable incision dz in each node will be half of the maximal DONWIND gradient
        # A minimal elevation diffrence threshold for incision is set. Below this value,
        # incision assumed to be zero (slope is VERY low).
        stable_incision_dz = (max_downwind_gradient * self._grid.dx) / 2  # topographic slope
        S[stable_incision_dz <= 0] = 0
        stable_incision_dz[
            stable_incision_dz <= 0] = np.inf  # set to infinity because slope is set to zero for this node (no erosion).
        E_tot = (detached_bedrock_rate_dz + detached_soil_rate_dz ) - deposited_sediments_dz_at_node
        if np.any(E_tot > 0):
            self._stable_dt_erosion = np.min(
                np.divide(
                    stable_incision_dz[E_tot > 0],
                    E_tot[E_tot > 0],
                )
            )
        else:
            self._stable_dt_erosion = np.inf


        # Condition 2.
        # Make sure deposition weight is not greater than what exist in the flow
        # The minimum dt based on this crietria will be 1 sec.
        if np.any(
                deposited_sediment_weights_at_node > 10 ** -10):  # some error that I allow for not get into small time steps all the time.
            self._stable_deposition_rate = \
                np.max([np.min(
                    np.divide(temp_sediment_load_weight_at_node_per_size,
                              deposited_sediment_weights_at_node,
                              where=deposited_sediment_weights_at_node > 10 ** -10,
                              out=np.ones_like(deposited_sediment_weights_at_node) * np.inf)), 1])
        else:
            self._stable_deposition_rate = np.inf


        # Condition 3.
        # Calc stable DEPOSITION depth:
        # Stable deposition dz in each node will be half of the maximal UPWIND gradient
        stable_deposition_dz = (np.abs(max_upwind_gradient) *
                                self._grid.dx) / 2  # Elevation diffrence of node to its UPWIND node
        stable_deposition_dz[
            stable_deposition_dz <= self._max_flipped_deposition_slope
            ] = self._max_flipped_deposition_slope * self._grid.dx

        net_deposition_dz = deposited_sediments_dz_at_node - (
                    detached_bedrock_rate_dz + detached_soil_rate_dz)  # bedrock erosion is not lowering the surface because its just convert 'bedrock' to 'soil'
        deposition_indices = np.where(
            (net_deposition_dz > 0.001))  # allow "small" piles of sediment to form (up to 0.001 [m] in height)
        if np.any(deposition_indices):
            self._stable_dt_deposition = np.min(
                np.divide(
                    stable_deposition_dz[deposition_indices],
                    net_deposition_dz[deposition_indices],
                    where=net_deposition_dz[deposition_indices] > 0.001,
                    out=np.ones_like(deposition_indices) * np.inf)
            )
        else:
            self._stable_dt_deposition = np.inf


        # Condition 4.
        # Avoid delivering more sediment than what existed in the upwind node
        sediment_load_weight_flux_at_node_per_size = self._sediment_load_flux_dzdt_at_node_per_size * self._grid.dx ** 2 * self._sigma * (
                1 - self._phi)
        sum_sediment_load_weight_flux_at_node = cfuncs_ErosionDeposition.grain_size_sum_at_node(
            sediment_load_weight_flux_at_node_per_size,
            np.zeros_like(self._zeros_at_node),
            np.shape(sediment_load_weight_flux_at_node_per_size))

        if np.any(sum_sediment_load_weight_flux_at_node < -self._min_sediment_load_weight_to_del):
            nodes_with_removed_sediment_load_mass = \
                np.where(sum_sediment_load_weight_flux_at_node < -self._min_sediment_load_weight_to_del)[0]

            if np.ndim(outlinks_fluxes_at_node) > 2:
                outfluxes_weights_at_node_per_size = np.sum(np.abs(outlinks_fluxes_at_node), axis=1)[
                                                     nodes_with_removed_sediment_load_mass, :]
            else:
                outfluxes_weights_at_node_per_size = np.abs(outlinks_fluxes_at_node)[
                                                     nodes_with_removed_sediment_load_mass, :]

            self._stable_dt_mass = np.divide(sediment_load_weight_at_node_per_size[
                                                 nodes_with_removed_sediment_load_mass, :],
                                             outfluxes_weights_at_node_per_size,
                                             out=np.ones_like(outfluxes_weights_at_node_per_size) * np.inf,
                                             where=outfluxes_weights_at_node_per_size > self._min_sediment_load_weight_to_del)

            self._stable_dt_mass = np.min(self._stable_dt_mass)
            if self._stable_dt_mass == 0:
                self._stable_dt_mass = np.inf
        else:
            self._stable_dt_mass = np.inf


        #self._stable_deposition_rate = np.inf  ## check if this is nesseary
        self._stable_dt = np.min((self._stable_dt_erosion,
                                  self._stable_dt_deposition,
                                  self._stable_dt_mass,
                                  self._stable_deposition_rate))


    def calc_rates(self):
        
        ## Initialized variabiles
        self._init_variables()

        ## Mapping
        self._map_upwind_downwind_nodes_to_links()

        ##
        self._calc_erosion_deposition()
        
        ##
        self._calc_load_flux()
        
        ## Calc the stable dt
        self._calc_stable_dt()

    
    def run_one_step_basic(self, dt=1):

        # Pointers
        grain_weights = self.grid.at_node['grains__weight']
        soil_depth = self._grid.at_node['soil__depth']
        bedrock = self._grid.at_node['bedrock__elevation']
        topo = self._grid.at_node['topographic__elevation']
        sediment_load_weight_at_node = self.grid.at_node['sediment_load__weight']
        sediment_load_weight_at_node_per_size = self._sediment_load_weight_at_node_per_size
        detached_soil_weight = self._detached_soil_weight
        detached_bedrock_weight = self._detached_bedrock_weight

        # Load fluxes per time step
        sediment_load_weight_flux_from_srrnds = self._sediment_load_flux_dzdt_at_node_per_size * self._grid.dx ** 2 * self._sigma * dt * (
                    1 - self._phi)  # NET flux after div. of suspended sediment at node

        ## Deposited from load
        deposited_sediment_weights_at_node = self._deposited_sediments_weights_at_node * dt  # Deposited weight of suspended sediment at the node
        deposited_sediment_weights_at_node[
            deposited_sediment_weights_at_node > sediment_load_weight_at_node_per_size] = \
        sediment_load_weight_at_node_per_size[
            deposited_sediment_weights_at_node > sediment_load_weight_at_node_per_size]

        ## Detached to load
        ## Update weight fluxes (deatched/deposited) at node
        local_sediment_weight_flux_at_node_per_size = (detached_soil_weight + detached_bedrock_weight) * dt  # Enrichment weight of suspended sediment at the node

        # Update load
        sediment_load_weight_at_node_per_size[:] = sediment_load_weight_at_node_per_size + (local_sediment_weight_flux_at_node_per_size + sediment_load_weight_flux_from_srrnds)
        sediment_load_weight_at_node_per_size[:] = sediment_load_weight_at_node_per_size - deposited_sediment_weights_at_node
        sediment_load_weight_at_node_per_size[sediment_load_weight_at_node_per_size < 0] = 0
        sediment_load_weight_at_node[:] = np.sum(sediment_load_weight_at_node_per_size, axis=1)


        # dz change in bedrock/soil layers
        detached_bedrock_rate_dz = self._detached_bedrock_rate_dz * dt
        detached_soil_weight = self._detached_soil_weight * dt



        if self._change_topo_flag:

            grain_weights[:] += deposited_sediment_weights_at_node[:]  - detached_soil_weight[:]
            grain_weights[grain_weights < 0] = 0

            soil_depth[:] = (np.sum(grain_weights, axis=1) / (self._sigma * self.grid.dx ** 2)) / (1 - self._phi)
            bedrock[:] -= detached_bedrock_rate_dz[:]
            topo[:] = soil_depth[:] + bedrock[:]
    