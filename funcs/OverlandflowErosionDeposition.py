"""
"""

import numpy as np
import scipy.constants
from landlab import Component
#from .cfuncs_erosion_deposition import cfuncs_ErosionDeposition
from funcs.cfuncs_erosion_deposition import cfuncs_ErosionDeposition
#import funcs.cfuncs_erosion_deposition
neg_WH = 10**-8
neg_GW = 10**-8
class OverlandflowErosionDeposition(Component):
    """Landlab component that simulates overland flow-driven erosion/deposition based on
    the approach presented by Foster and Meyer (1975) and Foster (1982).



    References:
    G. R. Foster and L. D. Meyer (1975) Mathematical simulation of upland erosion by fundamental erosion mechanics.
    p. 190-207. In Present and prospective technology for predicting sediment yields and sources.
    USDA-ARS. ARS-S-40. USDA

    Foster, G.R. (1982) Modeling the Erosion Process. In: Haan, C.T., Johnson, H.P. and Brakensiek, D.L., Eds.,
    Hydrologic Modeling of Small Watersheds, ASAE Monograph No. 5, American Society of Agricultural Engineers,
    St. Joseph, 297-380.
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
            "doc": "depth of water on the surface",
        },
        "surface_water__discharge": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m^3/s",
            "mapping": "link",
            "doc": "water discharge on the surface",
        },
        "topographic__elevation": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Land surface topographic elevation",
        },
        "bedrock__elevation": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "topographic elevation of the bedrock surface",
        },
        "soil__depth": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "soil layer depth",
        },
        "grains__weight": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "kg",
            "mapping": "node",
            "doc": "the weight of grains of different size in the soil layer",
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
            "doc": "Sediment flux (weight per unit time of sediment entering each node)",
        },
        "sediment__outflux": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m3/s",
            "mapping": "node",
            "doc": "Sediment flux (weight per unit time of sediment leaving each node)",
        },
    }

    def __init__(
            self,
            grid,
            fluid_density=1000.0,               # Fluid density [kg/m^3]
            g=scipy.constants.g,                # Gravitational acceleration [m/s^2]
            sigma=2000,                         # Soil density [kg/m^3]
            tau_crit=1.165,                     # Critical shear stress [pa]
            cover_depth_star=0.1,               # Characteristic soil roughness depth [m]
            phi=0.4,                            # Soil porosity [-]
            slope="topographic__slope",         # The slope used for calculate shear stress [-]
            bedrock_sediment_grainsizes=None,   # Bedrock layer grain sizes [-]
            R=1.65,                             # Submerged specific gravity [-]
            C1=18,                              # A constant with a theoretical value of 18 (Ferguson & Church 2004) [-]
            C2=0.4,                             # Asymptotic value of the drag coefficient (Ferguson & Church 2004) [-]
            v=9.2 * 10 ** -7,                   # The kinematic viscosity of the fluid [kg/m/s]
            alpha=0.045,                        # 0.045 From Komar, P. D. (1987).
            beta=-0.68,                         # -0.68 From Komar, P. D. (1987).
            kr=0.0002,                          # Erodibility coefficient [s/m]
            max_flipped_deposition_dz=0.01,     # Allow inverse of topography up to certain dz [m]
            depression_depth=0.0055,            # Characteristic depression depth for calculating flow width [m]
            roughness_correction = 1,           # Roughness correction for shear stress [-]
            change_topo_flag=True,  # A flag the allow to run simulation without changing the topography

    ):

        super().__init__(grid)

        assert slope in grid.at_node
        self._slope = slope
        self.initialize_output_fields()
        self._change_topo_flag = change_topo_flag

        # Grid parameters
        self._zeros_at_link = self._grid.zeros(at="link")
        self._zeros_at_node = self._grid.zeros(at="node")
        self._xy_spacing = np.array(self.grid.spacing)
        self._grid_shape = np.array(self.grid.shape)
        self._nodes_flatten = grid.nodes.flatten().astype('int')
        self._nodes = np.shape(self._grid.nodes)[0] * np.shape(self._grid.nodes)[1]
        self._inactive_links = grid.status_at_link == grid.BC_LINK_IS_INACTIVE
        self._active_links = ~(grid.status_at_link == grid.BC_LINK_IS_INACTIVE)
        self._links_array = np.arange(0, np.size(self._zeros_at_link)).tolist()
        self._active_links_ids = np.array(self._links_array)[self._active_links]
        self._zeros_at_node_for_fractions = np.zeros((self._nodes, np.shape(self._grid.at_node['grains__weight'])[1]))
        self._zeros_at_links_for_fractions = np.zeros(
            (np.size(self._zeros_at_link), np.shape(self._grid.at_node['grains__weight'])[1]))
        self._n_grain_sizes = np.shape(self._grid.at_node['grains__weight'])[1]

        # Vars
        self._g = g
        self._rho = fluid_density
        self._sigma = sigma
        self._phi = phi
        self._cover_depth_star = cover_depth_star
        self._roughness_correction = roughness_correction
        self._R = R
        self._C1 = C1
        self._C2 = C2
        self._v = v
        self._SG = self._sigma / self._rho
        self._alpha = alpha
        self._beta = beta
        self._kr = kr

        # Thresholds for stability criteria
        self._min_sediment_load_weight_to_del = neg_GW
        self._max_flipped_deposition_dz = max_flipped_deposition_dz
        self._depression_depth = depression_depth

        # Fields
        self._bedrock_sediment_grainsizes = bedrock_sediment_grainsizes
        self._sediment_load_flux_dzdt_at_node_per_size = self._zeros_at_node_for_fractions
        self._weight_flux_at_link = np.zeros(
            (np.size(self._zeros_at_link), np.shape(self._grid.at_node['grains__weight'])[1]))
        self._sediment_load_concentration_at_link = np.zeros(
            (np.size(self._zeros_at_link), np.shape(self._grid.at_node['grains__weight'])[1]))
        self._DR = np.zeros((self._nodes, np.shape(self._grid.at_node['grains__weight'])[1]))
        self._tau_crit = np.zeros((self._nodes, np.shape(self._grid.at_node['grains__weight'])[1]))
        self._tau_crit[:] = tau_crit
        self._outlinks_fluxes_at_node = np.zeros((self._nodes, np.shape(self._grid.at_node['grains__weight'])[1]))
        self._inlinks_fluxes_at_node = np.zeros((self._nodes,  np.shape(self._grid.at_node['grains__weight'])[1]))
        self._c_si = np.zeros((self._nodes, np.shape(self._grid.at_node['grains__weight'])[1]))
        self._c_kg = np.zeros((self._nodes, np.shape(self._grid.at_node['grains__weight'])[1]))
        self._grain_fractions_at_node = np.zeros((self._nodes, np.shape(self._grid.at_node['grains__weight'])[1]))
        self._sediment_load_size_fractions_at_node = np.zeros(
            (self._nodes, np.shape(self._grid.at_node['grains__weight'])[1]))
        self._TC = np.zeros((self._nodes, np.shape(self._grid.at_node['grains__weight'])[1]))
        self._bedrock_grain_fractions = self._grid.at_node["bed_grains__proportions"][self._grid.core_nodes[0]]
        self._bedrock_median_size = self._grid.at_node["grains_classes__size"][self._grid.core_nodes[0]][
            int(np.argwhere(np.cumsum(self._bedrock_grain_fractions) >= 0.5)[0])]
        if self._bedrock_sediment_grainsizes == None:
            self._bedrock_sediment_grainsizes = self._grid.at_node["grains_classes__size"][self._grid.core_nodes[0]]

        self._sediment_load_weight_flux_at_link = np.zeros(
            (np.shape(self._zeros_at_link)[0], np.shape(self._grid.at_node['grains__weight'])[1]))
        self._shape_links_gs = np.shape(self._sediment_load_weight_flux_at_link)

        self._sediment_load_weight_at_node_per_size = np.zeros(
            (np.shape(self._zeros_at_node)[0], np.shape(self._grid.at_node['grains__weight'])[1]))
        self._deposited_sediments_dz_at_node = np.zeros_like(self._zeros_at_node)
        self._detached_bedrock_rate_dz = np.zeros_like(self._zeros_at_node)
        self._detached_soil_rate_dz = np.zeros_like(self._zeros_at_node)
        self._detached_bedrock_weight = np.zeros(
            (np.shape(self._zeros_at_node)[0], np.shape(self._grid.at_node['grains__weight'])[1]))
        self._detached_soil_weight = np.zeros(
            (np.shape(self._zeros_at_node)[0], np.shape(self._grid.at_node['grains__weight'])[1]))
        self._deposited_sediments_weights_at_node = np.zeros(
            (np.shape(self._zeros_at_node)[0], np.shape(self._grid.at_node['grains__weight'])[1]))
        self._vs = np.zeros((1, np.shape(self._grid.at_node['grains__weight'])[
            1]))

        # Calc settling velocity for all grain size classes
        self._calc_settling_velocity_per_size()



    @property
    def settling_velocities(self):
        return self._vs

    @property
    def sediment_load(self):
        return self._sediment_load_size_fractions_at_node

    @property
    def shear_stress(self):
        return self._tau_s


    def _calc_settling_velocity_per_size(self):
        # Based on: https://pubs.geoscienceworld.org/sepm/jsedres/article/74/6/933/99413/A-Simple-Universal-Equation
        # -for-Grain-Settling
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
        self._sediment_load_weight_flux_at_link.fill(0.0)
        self._c_si.fill(1.0)
        self._c_kg.fill(1.0)
        self._grain_fractions_at_node.fill(0.0)
        self._sediment_load_size_fractions_at_node.fill(0.0)
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

        # Limit time step reduction to 1 sec (consider to apply if run time is very slow)
        indices_to_correct_flux = np.where(outlinks_fluxes_at_node > sediment_load_weight_at_node_per_size)
        if np.any(indices_to_correct_flux):
            (weight_flux_at_link,
             outlinks_fluxes_at_node,
             inlinks_fluxes_at_node,
             total_outflux_at_node,
             total_influx_at_node
             ) = self._fix_outfluxes(indices_to_correct_flux,
                                     outlinks_fluxes_at_node,
                                     sediment_load_weight_at_node_per_size,
                                     outlinks_at_node,
                                     upwind_node_ids_at_link,
                                     downwind_node_ids_at_link,
                                     weight_flux_at_link,
                                     shape)

        self._grid.at_node['sediment__influx'][:] = total_influx_at_node
        self._grid.at_node['sediment__outflux'][:] = total_outflux_at_node

        return weight_flux_at_link, outlinks_fluxes_at_node, inlinks_fluxes_at_node


    def _fix_outfluxes(self, indices_to_correct_flux,
                       outlinks_fluxes_at_node,
                       sediment_load_weight_at_node_per_size,
                       outlinks_at_node,
                       upwind_node_ids_at_link,
                       downwind_node_ids_at_link,
                       weight_flux_at_link,
                       shape):

        ratios = np.divide(sediment_load_weight_at_node_per_size[indices_to_correct_flux],
                           outlinks_fluxes_at_node[indices_to_correct_flux])

        for i, (n, gs) in enumerate(zip(indices_to_correct_flux[0], indices_to_correct_flux[1])):
            out_links = self._grid.links_at_node[n, :][outlinks_at_node[n, :]]
            weight_flux_at_link[out_links, gs] *= ratios[i, np.newaxis]  # Update the weight flux at link.

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

        return (weight_flux_at_link,
                outlinks_fluxes_at_node,
                inlinks_fluxes_at_node,
                total_outflux_at_node,
                total_influx_at_node)

    def _calc_dzdt(self, size_class, dt=1):

        size_class = int(size_class)
        dt = dt
        xy_spacing = self._xy_spacing
        shape = self._grid_shape
        out = np.zeros_like(self._zeros_at_node)
        dzdt = cfuncs_ErosionDeposition.calc_flux_div_at_node(shape,
                                                              xy_spacing,
                                                              self._sediment_load_weight_flux_at_link[:,
                                                              size_class] * dt,
                                                              out)

        dzdt = -dzdt
        return dzdt

    def _calc_sediments_flux_at_link(self):

        # Calc net sediment load flux
        size_class = np.where(np.any(self._sediment_load_weight_flux_at_link, axis=0))[0].tolist()
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

        # Pointers
        c_si_kg = self._c_kg
        surface_water__depth_at_node = self.grid.at_node['surface_water__depth']
        S = self._grid.at_node[self._slope]
        median_sizes = np.copy(self._grid.at_node['median_size__weight'])
        median_sizes[median_sizes == 0] = self._bedrock_median_size

        # Get outflux water discharge
        shape = (np.size(self._active_links_ids))
        q_at_node = cfuncs_ErosionDeposition.sum_out_discharge(self._upwind_node_ids_at_link,
                                                               np.abs(self._grid.at_link['surface_water__discharge']),
                                                               self._active_links_ids,
                                                               np.zeros_like(self._zeros_at_node),
                                                               shape)

        # Calc shear stress at node.
        self._tau_s = self._rho * self._g * S * surface_water__depth_at_node * self._roughness_correction

        # Flow width is calculated according to the fraction of the cell covered by water which is approximated by the
        # relationship between surface runoff height and maximum depression capacity (5.5 mm for shrub) (Nunes et 
        # al., 2005, CATENA)
        flow_width = np.divide(surface_water__depth_at_node,
                               self._depression_depth)   # depression depth in meters
        flow_width[
            flow_width > self._grid.dx] = self._grid.dx  # greater than 1 -> water depth is over the depression
        # depth so flow width is set to the grid node width

        # Calculation of transport capacity at node
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

        # Calculation of Dc (detachment rate)
        self._Dc = cfuncs_ErosionDeposition.calc_Dc(self._tau_s,
                                                    self._tau_crit,
                                                    self._grid.core_nodes,
                                                    np.zeros_like(self._zeros_at_node_for_fractions),
                                                    self._kr,
                                                    shape)

        # Get the total weight flux
        CQ = cfuncs_ErosionDeposition.calc_CQ(
            c_si_kg,
            np.zeros_like(self._zeros_at_node_for_fractions),
            q_at_node,
            self._grid.core_nodes,
            shape,
            int(self._grid.dx))

        # Calculation of net erosion/deposition at node
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

        # Pointers
        water_surface_grad_at_link = self._grid.calc_grad_at_link('water_surface__elevation')
        sediment_load_concentration_at_link = self._sediment_load_concentration_at_link
        weight_flux_at_link = self._weight_flux_at_link
        sediment_load_weight_at_node_per_size = self._sediment_load_weight_at_node_per_size
        upwind_node_ids_at_link = self._upwind_node_ids_at_link
        downwind_node_ids_at_link = self._downwind_node_ids_at_link
        surface_water__depth_at_node = self.grid.at_node['surface_water__depth']
        surface_water__depth_at_node[surface_water__depth_at_node < neg_WH] = neg_WH
        surface_water__depth_at_node_expand = np.expand_dims(surface_water__depth_at_node, -1)
        q_water_at_link = self.grid.at_link[
            'surface_water__discharge']
        q_water_at_link[self._inactive_links] = 0

        # Weight concentration at link
        sediment_load_concentration_at_link[:] = np.divide(sediment_load_weight_at_node_per_size[
                                                           upwind_node_ids_at_link, :],
                                                           surface_water__depth_at_node_expand[
                                                           upwind_node_ids_at_link,
                                                           :] * self._grid.dx * self._grid.dx)
        # Find the outlinks for each node.
        outlinks_at_node = self.grid.link_at_node_is_downwind(water_surface_grad_at_link)

        # Calc weight flux at link
        shape = [np.size(self._active_links_ids), self._n_grain_sizes]
        weight_flux_at_link[:] = cfuncs_ErosionDeposition.calc_flux_at_link_per_size(q_water_at_link,
                                                                                     sediment_load_concentration_at_link,
                                                                                     self._grid.active_links,
                                                                                     np.zeros_like(weight_flux_at_link),
                                                                                     shape)
        weight_flux_at_link = np.abs(weight_flux_at_link)

        # Get the in/out sediment load weight flux at NODE
        weight_flux_at_link[:], outlinks_fluxes_at_node, inlinks_fluxes_at_node = self._calc_inout_fluxes_at_node(
            upwind_node_ids_at_link,
            downwind_node_ids_at_link,
            weight_flux_at_link,
            sediment_load_weight_at_node_per_size,
            outlinks_at_node)

        self._outlinks_fluxes_at_node[:] = outlinks_fluxes_at_node
        self._inlinks_fluxes_at_node[:] = inlinks_fluxes_at_node

        # Get the sediment load weight flux at LINK
        shape = np.copy(self._shape_links_gs)
        self._sediment_load_weight_flux_at_link[:] = cfuncs_ErosionDeposition.calc_flux_at_link(self._grid.dx,
                                                                                                self._sigma,
                                                                                                self._phi,
                                                                                                np.abs(
                                                                                                    weight_flux_at_link),
                                                                                                -np.sign(
                                                                                                    water_surface_grad_at_link),
                                                                                                self._sediment_load_weight_flux_at_link,
                                                                                                shape)

        # Calculate sediment-load flux at link
        self._calc_sediments_flux_at_link()

    def _map_upwind_downwind_nodes_to_links(self):

        # Map upwind/downwind node id to links
        self._upwind_node_ids_at_link = self._grid.map_value_at_max_node_to_link('water_surface__elevation',
                                                                           self._nodes_flatten).astype('int')
        self._downwind_node_ids_at_link = self._grid.map_value_at_min_node_to_link('water_surface__elevation',
                                                                             self._nodes_flatten).astype('int')

    def _calc_erosion_deposition(self, ):

        # Pointers
        surface_water__depth_at_node = self.grid.at_node['surface_water__depth']
        surface_water__depth_at_node[surface_water__depth_at_node < neg_WH] = neg_WH
        surface_water__depth_at_node_expand = np.expand_dims(surface_water__depth_at_node, -1)
        soil_depth = self._grid.at_node['soil__depth']
        grain_weight_at_node = self.grid.at_node['grains__weight']
        q_water_at_link = self.grid.at_link[
            'surface_water__discharge']  # surface water discharge units are m**3/s -> From OverlandFlow component
        q_water_at_link[self._inactive_links] = 0
        deposited_sediment_weights_at_node = self._deposited_sediments_weights_at_node
        sediment_load_weight_at_node_per_size = self._sediment_load_weight_at_node_per_size
        deposited_sediments_dz_at_node = self._deposited_sediments_dz_at_node
        detached_bedrock_rate_dz = self._detached_bedrock_rate_dz
        detached_soil_rate_dz = self._detached_soil_rate_dz
        detached_bedrock_weight = self._detached_bedrock_weight
        detached_soil_weight = self._detached_soil_weight

        c_si = self._c_si
        c_kg = self._c_kg
        grain_fractions_at_node = self._grain_fractions_at_node
        sediment_load_size_fractions_at_node = self._sediment_load_size_fractions_at_node
        grain_weight_at_node[grain_weight_at_node < neg_GW] = neg_GW

        # Calc soil exponent
        soil_e_expo = (1 - (np.exp(
            (-soil_depth) / self._cover_depth_star))
                       )

        # Sediment load total volume
        temp_sediment_load_weight_at_node_per_size = np.copy(sediment_load_weight_at_node_per_size)
        temp_sediment_load_weight_at_node_per_size[temp_sediment_load_weight_at_node_per_size < 0] = 0
        # Convert to volume with no porosity correction
        temp_sediment_load_volume_at_node_per_size = temp_sediment_load_weight_at_node_per_size / self._sigma
        temp_sediment_load_volume_at_node_per_size[temp_sediment_load_volume_at_node_per_size < 0] = 0

        # Concentrations (volumetric and weight)
        c_si[:] = np.divide(temp_sediment_load_volume_at_node_per_size,
                            surface_water__depth_at_node_expand * self.grid.dx ** 2,
                            out=np.ones_like(temp_sediment_load_volume_at_node_per_size))

        # Weight concentration
        c_kg[:] = np.divide(temp_sediment_load_weight_at_node_per_size,
                            surface_water__depth_at_node_expand * self.grid.dx ** 2,
                            out=np.ones_like(temp_sediment_load_volume_at_node_per_size))

        # Calc weight fraction of all size classes
        temp_sediment_load_weight_at_node_per_size[
            temp_sediment_load_weight_at_node_per_size < neg_GW] = neg_GW
        sediment_load_size_fractions_at_node[:] = cfuncs_ErosionDeposition.calc_concentration(
            np.ones_like(temp_sediment_load_weight_at_node_per_size),
            temp_sediment_load_weight_at_node_per_size,
            np.sum(temp_sediment_load_weight_at_node_per_size, 1),
            np.shape(temp_sediment_load_weight_at_node_per_size)
        )

        sum_c_si = cfuncs_ErosionDeposition.grain_size_sum_at_node(c_si,
                                                                   np.zeros_like(self._zeros_at_node),
                                                                   np.shape(c_si))

        # Calculation of erosion/deposition at node
        self._calc_DR()
        # If the concentration is greater than 1, all the sediments in the load needs to be deposited
        self._DR[sum_c_si >= 1, :] = -np.inf  # DR returns in units of kg/(m^2*s)

        # Calc net change for both layer
        core_nodes = self._grid.core_nodes
        factor_convert_weight_to_dz_c = self._sigma * (1 - self._phi) * self._grid.dx ** 2
        factor_convert_weight_to_dz_bedrock_c = self._sigma * self._grid.dx ** 2
        shape = (np.size(core_nodes), np.shape(self._zeros_at_node_for_fractions)[1])
        dx_c = self._grid.dx

        # Calc weight concentration at node
        grain_fractions_at_node[:] = cfuncs_ErosionDeposition.calc_concentration(np.ones_like(grain_weight_at_node),
                                                                                 grain_weight_at_node,
                                                                                 np.sum(grain_weight_at_node, axis=1),
                                                                                 np.shape(grain_weight_at_node)
                                                                                 )
        # Calc Erosion/Deposition at node
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
                                                                                               sediment_load_size_fractions_at_node,
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

        # Pointers
        max_downwind_gradient = self._grid.at_node['downwind__link_gradient']
        max_upwind_gradient = self._grid.at_node['upwind__link_gradient']
        S = self._grid.at_node[self._slope]
        detached_bedrock_rate_dz = self._detached_bedrock_rate_dz
        detached_soil_rate_dz = self._detached_soil_rate_dz
        deposited_sediments_dz_at_node = self._deposited_sediments_dz_at_node
        deposited_sediment_weights_at_node = self._deposited_sediments_weights_at_node
        temp_sediment_load_weight_at_node_per_size = np.copy(self._sediment_load_weight_at_node_per_size)
        temp_sediment_load_weight_at_node_per_size[temp_sediment_load_weight_at_node_per_size < 0] = 0
        sediment_load_weight_at_node_per_size = self._sediment_load_weight_at_node_per_size
        outlinks_fluxes_at_node = self._outlinks_fluxes_at_node

        # Condition 1: Stable incision dz
        # Stable incision dz in each node will be half of the maximal DOWNWIND gradient
        # A minimal elevation difference threshold for incision is set. Below this value,
        # incision assumed to be zero (slope is VERY low).
        stable_incision_dz = (max_downwind_gradient * self._grid.dx) / 2  # topographic slope
        S[stable_incision_dz <= 0] = 0
        stable_incision_dz[
            stable_incision_dz <= 0] = np.inf  # set to infinity because slope is set to zero for this node (no erosion).
        E_tot = (detached_bedrock_rate_dz + detached_soil_rate_dz) - deposited_sediments_dz_at_node
        if np.any(E_tot > 0):
            self._stable_dt_erosion = np.min(
                np.divide(
                    stable_incision_dz[E_tot > 0],
                    E_tot[E_tot > 0],
                )
            )
        else:
            self._stable_dt_erosion = np.inf

        # Condition 2. Deposition weight
        # Make sure the deposited weight is not greater than what exist in the flow
        if np.any(
                deposited_sediment_weights_at_node > neg_GW):  # some error that I allow for not get into small
            # time steps.
            self._stable_deposition_rate = \
                np.max([np.min(
                    np.divide(temp_sediment_load_weight_at_node_per_size,
                              deposited_sediment_weights_at_node,
                              where=deposited_sediment_weights_at_node > neg_GW,
                              out=np.ones_like(deposited_sediment_weights_at_node) * np.inf)), 1])
        else:
            self._stable_deposition_rate = np.inf

        # Condition 3. Deposition depth
        # Stable deposition dz in each node will be half of the maximal UPWIND gradient
        stable_deposition_dz = (np.abs(max_upwind_gradient) *
                                self._grid.dx) / 2  # Elevation difference of node to its UPWIND node
        stable_deposition_dz[
            stable_deposition_dz <= self._max_flipped_deposition_dz
            ] = self._max_flipped_deposition_dz * self._grid.dx

        net_deposition_dz = deposited_sediments_dz_at_node - (
                detached_bedrock_rate_dz + detached_soil_rate_dz)

        deposition_indices = np.where(
            (net_deposition_dz > self._max_flipped_deposition_dz))  # allow "small" piles of sediment to form
        if np.any(deposition_indices):
            self._stable_dt_deposition = np.min(
                np.divide(
                    stable_deposition_dz[deposition_indices],
                    net_deposition_dz[deposition_indices],
                    where=net_deposition_dz[deposition_indices] > self._max_flipped_deposition_dz,
                    out=np.ones_like(deposition_indices) * np.inf)
            )
        else:
            self._stable_dt_deposition = np.inf

        # Condition 4: Avoid delivering more sediment than what existed in the upwind node
        sediment_load_weight_flux_at_node_per_size = (self._sediment_load_flux_dzdt_at_node_per_size * 
                                                      self._grid.dx ** 2 * self._sigma * (1 - self._phi))
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

        self._stable_dt = np.min((self._stable_dt_erosion,
                                  self._stable_dt_deposition,
                                  self._stable_dt_mass,
                                  self._stable_deposition_rate))

    def calc_rates(self):

        # Initialized variables
        self._init_variables()

        # Mapping
        self._map_upwind_downwind_nodes_to_links()

        # Calculate the rate of erosion/deposition
        self._calc_erosion_deposition()

        # Calculate the load flux from in/out discharge
        self._calc_load_flux()

        # Calculate the stable dt based on the updated rates
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
        core_nodes = self._grid.core_nodes

        # Convert load flux from dz to weight
        sediment_load_weight_flux_from_srrnds = (self._sediment_load_flux_dzdt_at_node_per_size *
                                                 self._grid.dx**2 * 
                                                 self._sigma * dt *
                                                 (1 - self._phi))  # NET flux after div. of sediment load at node

        # Deposited from load
        deposited_sediment_weights_at_node = self._deposited_sediments_weights_at_node * dt 
        deposited_sediment_weights_at_node[
            deposited_sediment_weights_at_node > sediment_load_weight_at_node_per_size] = \
            sediment_load_weight_at_node_per_size[
                deposited_sediment_weights_at_node > sediment_load_weight_at_node_per_size]

        # Update detached weights (soil and bedrock) at node
        local_sediment_weight_flux_at_node_per_size = (detached_soil_weight + detached_bedrock_weight) * dt  # 

        # Update net sediment load at node according to local detachment soil and bedrock
        sediment_load_weight_at_node_per_size[:] = sediment_load_weight_at_node_per_size + (
                    local_sediment_weight_flux_at_node_per_size + sediment_load_weight_flux_from_srrnds)

        # Update net sediment load after deposition from the flow
        sediment_load_weight_at_node_per_size[
        :] = sediment_load_weight_at_node_per_size - deposited_sediment_weights_at_node

        sediment_load_weight_at_node_per_size[sediment_load_weight_at_node_per_size < 0] = 0
        sediment_load_weight_at_node[:] = np.sum(sediment_load_weight_at_node_per_size, axis=1)

        # dz change in bedrock/soil layers
        detached_bedrock_rate_dz = self._detached_bedrock_rate_dz * dt
        detached_soil_weight = self._detached_soil_weight * dt

        if self._change_topo_flag:
            grain_weights[:] += deposited_sediment_weights_at_node[:] - detached_soil_weight[:]
            grain_weights[grain_weights < 0] = 0

            soil_depth[core_nodes] = (np.sum(grain_weights[core_nodes,:], axis=1) / (self._sigma * self.grid.dx ** 2)) / (1 - self._phi)
            bedrock[core_nodes] -= detached_bedrock_rate_dz[core_nodes]
            topo[core_nodes] = soil_depth[core_nodes] + bedrock[core_nodes]
