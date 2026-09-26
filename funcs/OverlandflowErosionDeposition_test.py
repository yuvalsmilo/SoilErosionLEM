"""
"""

import numpy as np
import scipy.constants
from landlab import Component
import cfuncs_ErosionDeposition


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
        "suspended__sediments_masss": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "kg",
            "mapping": "node",
            "doc": "mass of suspended sediment per grain size at the node",
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

        "excess___stress": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "Pa",
            "mapping": "node",
            "doc": "Excess shear stress",
        },
    }

    def __init__(
            self,
            grid,
            fluid_density=1000.0,
            g=scipy.constants.g,
            sigma=2650,
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
            detachment_model='shear',
            # 'shear' (Dc = kr*(tau_s - tau_crit)) or 'stream_power' (Dc = k_omega * rho*g*S*q, RHEM V2.3)
            k_omega=7.74 * 10 ** -3,
            # stream-power erodibility coefficient [s^2/m^2], used when detachment_model='stream_power'
            Cv_max=100.75,
            # sediment/water volume ratio at which transport capacity is fully choked off (~0.6-0.65 packing fraction by volume)
            Cv_taper_exponent=2,  # shape of the capacity taper as concentration approaches Cv_max
            DR_relaxation=0.5,
            # under-relaxation on DR (1=no damping/original behavior, lower=more damping of the detachment/deposition bang-bang oscillation)
            q_at_node_relaxation=0.1,
            # exponential smoothing on q_at_node (1=no smoothing/raw discharge, lower=more smoothing of the momentum-solution chatter)
            Kv=3 * 10 ** -8,
            max_flipped_deposition_slope=0.01,  # [m/m] -- allow inverse of topography up to certain slope
            depression_depth=0.0055,
            # Charteristic surface depression depth/height [m]. Used to calculate flow 'width' in each cell
            omega_veg=0.5,
            # Vegetation-erosion dynamics from here:   https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2004JF000249
            veg_roughness_reference=0.1,
            veg_cover_reference=0.6,
            soil_roughness=0.025,
            veg_flag=0,  # 1 == Using KV,  2 == Using ABS(E)
            veg_cover_fraction=1,
            ft=1,
    ):

        super().__init__(grid)

        assert slope in grid.at_node
        self.initialize_output_fields()
        self._zeros_at_link = self._grid.zeros(at="link")
        self._zeros_at_node = self._grid.zeros(at="node")
        self._nodes_at_cell = self._grid.map_node_to_cell(self._grid.nodes.flatten()).astype(int)

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
        assert detachment_model in ('shear', 'stream_power'), \
            "detachment_model must be 'shear' or 'stream_power'"
        self._detachment_model = detachment_model
        self._k_omega = k_omega
        self._Cv_max = Cv_max
        self._Cv_taper_exponent = Cv_taper_exponent
        assert 0 < DR_relaxation <= 1, "DR_relaxation must be in (0, 1]"
        self._DR_relaxation = DR_relaxation
        assert 0 < q_at_node_relaxation <= 1, "q_at_node_relaxation must be in (0, 1]"
        self._q_at_node_relaxation = q_at_node_relaxation
        self._q_at_node_smoothed = None  # lazily initialized on first _calc_DR call
        self._ft = ft
        self._veg_cover_fraction = veg_cover_fraction

        self._Kv = Kv
        self._bedrock_sediment_grainsizes = bedrock_sediment_grainsizes
        self._nodes_flatten = grid.nodes.flatten().astype('int')
        self._nodes = np.shape(self._grid.nodes)[0] * np.shape(self._grid.nodes)[1]

        self._inactive_links = grid.status_at_link == grid.BC_LINK_IS_INACTIVE
        self._active_links = ~(grid.status_at_link == grid.BC_LINK_IS_INACTIVE)
        self._links_array = np.arange(0, np.size(self._zeros_at_link)).tolist()
        self._active_links_ids = np.array(self._links_array)[self._active_links]

        # Landlab collapses a per-node field down to 1D (n_nodes,) - instead
        # of (n_nodes, n_sizes) - whenever there is only a single grain size
        # class, regardless of how it was created. np.shape(...)[1] then
        # raises IndexError since there is no axis 1. Guard against that
        # here (n_sizes == 1 in that case) rather than assuming 2D.
        grains_mass_field = self._grid.at_node['grains__mass']
        if np.ndim(grains_mass_field) > 1:
            n_grain_sizes = np.shape(grains_mass_field)[1]
        else:
            n_grain_sizes = 1

        self._n_grain_sizes = n_grain_sizes
        self._zeros_at_node_for_fractions = np.zeros((self._nodes, n_grain_sizes))
        self._zeros_at_links_for_fractions = np.zeros(
            (np.size(self._zeros_at_link), n_grain_sizes))
        self._mass_flux_at_link = np.zeros(
            (np.size(self._zeros_at_link), n_grain_sizes))
        self._suspended__sediments_concentration_at_link = np.zeros(
            (np.size(self._zeros_at_link), n_grain_sizes))
        self._DR = np.zeros((self._nodes, n_grain_sizes))

        self._tau_crit = np.zeros((self._nodes, n_grain_sizes))
        self._tau_crit_s = np.zeros((self._nodes, n_grain_sizes))
        self._tau_crit[:] = tau_crit
        self._tau_crit_s[:] = tau_crit
        self._suspended_dzdt_at_node_per_size = self._zeros_at_node_for_fractions
        self._inactive_links = grid.status_at_link == grid.BC_LINK_IS_INACTIVE

        self._outlinks_fluxes_at_node = np.zeros((self._nodes, n_grain_sizes))
        self._inlinks_fluxes_at_node = np.zeros((self._nodes, n_grain_sizes))
        self._c_si = np.zeros((self._nodes, n_grain_sizes))
        self._c_kg = np.zeros((self._nodes, n_grain_sizes))
        self._sum_c_si = np.zeros(self._nodes)  # total sediment/water volume ratio per node, used to taper TC
        self._grain_fractions_at_node = np.zeros((self._nodes, n_grain_sizes))
        self._suspended_fraction_at_node = np.zeros((self._nodes, n_grain_sizes))
        self._TC = np.zeros((self._nodes, n_grain_sizes))
        self._change_topo_flag = change_topo_flag
        self._xy_spacing = np.array(self.grid.spacing)
        self._grid_shape = np.array(self.grid.shape)
        self._min_suspended_mass_to_del = 10 ** -10

        self._max_flipped_deposition_slope = max_flipped_deposition_slope
        self._depression_depth = depression_depth
        # Same field-squeeze issue throughout this file: with a single grain
        # size, bed_grains__proportions loses its second axis too. Restore
        # a 2D view (not a copy - in-place writes elsewhere still propagate)
        # so every later use of self._bedrock_grain_fractions (calc_DR,
        # calc_detached_deposited, ...) sees consistent (n_nodes, n_sizes)
        # shape regardless of how many grain sizes there are.
        self._bedrock_grain_fractions = self._grid.at_node["bed_grains__proportions"]
        if np.ndim(self._bedrock_grain_fractions) == 1:
            self._bedrock_grain_fractions = self._bedrock_grain_fractions.reshape(-1, 1)
        # Same field-squeeze issue as above: with a single grain size class,
        # grains_classes__size[core_nodes[0]] comes back as a bare scalar
        # (not a length-1 array), so indexing it with [median_idx] raises
        # "invalid index to scalar variable". With only one grain size, that
        # scalar trivially *is* the median, so use it directly in that case.
        grain_sizes_at_core0 = self._grid.at_node["grains_classes__size"][self._grid.core_nodes[0]]
        median_idx = int(np.argwhere(np.cumsum(self._bedrock_grain_fractions[grid.core_nodes[0]]) >= 0.5)[0])
        if np.ndim(grain_sizes_at_core0) > 0:
            self._bedrock_median_size = grain_sizes_at_core0[median_idx]
        else:
            self._bedrock_median_size = grain_sizes_at_core0
        if self._bedrock_sediment_grainsizes == None:
            self._bedrock_sediment_grainsizes = self._grid.at_node["grains_classes__size"][self._grid.core_nodes[0]]

        self._suspended__sediments_mass_at_link = np.zeros(
            (np.shape(self._zeros_at_link)[0], n_grain_sizes))
        self._suspended__sediments_flux_at_link = np.zeros(
            (np.shape(self._zeros_at_link)[0], n_grain_sizes))

        self._suspended_sediment_mass_at_node_per_size = np.zeros(
            (np.shape(self._zeros_at_node)[0], n_grain_sizes))
        self._deposited_suspended_sediments_dz_at_node = self._grid.zeros(at="node")
        self._local_sediment_mass_flux_at_node_per_size = np.zeros(
            (np.shape(self._zeros_at_node)[0], n_grain_sizes))
        self._detached_bedrock_rate_dz = np.zeros_like(self._zeros_at_node)
        self._detached_soil_rate_dz = np.zeros_like(self._zeros_at_node)

        self._detached_bedrock_mass = np.zeros(
            (np.shape(self._zeros_at_node)[0], n_grain_sizes))
        self._detached_soil_mass = np.zeros(
            (np.shape(self._zeros_at_node)[0], n_grain_sizes))
        self._deposited_suspended_sediments_masss_at_node = np.zeros(
            (np.shape(self._zeros_at_node)[0], n_grain_sizes))

        self._vs = np.zeros(
            (1, n_grain_sizes))  # ratio of near-bed sediment concentration to the vertically averaged concentration
        # Same field-squeeze issue as above: with a single grain size class
        # this comes back as a bare scalar, which isn't iterable. Wrap with
        # atleast_1d - a no-op for the normal (already-array) multi-size
        # case, and turns a lone scalar into a 1-element array so the loop
        # below still runs once as intended.
        grain_sizes_for_settling = np.atleast_1d(self._grid.at_node["grains_classes__size"][self._grid.core_nodes[0]])
        for i, g_size in enumerate(grain_sizes_for_settling):
            self._vs[0, i] = np.divide(
                (self._R * self._g * g_size ** 2),
                (self._C1 * self._v + (0.75 * self._C2 * self._R * self._g * (g_size) ** 3) ** 0.5)
            )  # from here: https://pubs.geoscienceworld.org/sepm/jsedres/article/74/6/933/99413/A-Simple-Universal-Equation-for-Grain-Settling

        ## Vegetation-erosion parameters
        self._veg_flag = veg_flag
        self._omega_veg = omega_veg
        self._veg_roughness_reference = veg_roughness_reference
        self._veg_cover_reference = veg_cover_reference
        self._soil_roughness = soil_roughness

    def _init_variables(self):

        self._detached_soil_mass.fill(0.0)
        self._detached_bedrock_mass.fill(0.0)
        self._detached_bedrock_rate_dz.fill(0.0)
        self._detached_soil_rate_dz.fill(0.0)
        self._suspended_dzdt_at_node_per_size.fill(0.0)
        self._deposited_suspended_sediments_masss_at_node.fill(0.0)
        self._deposited_suspended_sediments_dz_at_node.fill(0.0)
        self._local_sediment_mass_flux_at_node_per_size.fill(0.0)
        self._outlinks_fluxes_at_node.fill(0.0)
        self._inlinks_fluxes_at_node.fill(0.0)
        self._suspended__sediments_concentration_at_link.fill(0.0)
        self._grid.at_node['sediment__influx'].fill(0.0)
        self._grid.at_node['sediment__outflux'].fill(0.0)
        self._suspended__sediments_flux_at_link.fill(0.0)
        self._c_si.fill(1.0)
        self._c_kg.fill(1.0)
        self._grain_fractions_at_node.fill(0.0)
        self._suspended_fraction_at_node.fill(0.0)
        self._mass_flux_at_link.fill(0.0)

    def _calc_inout_fluxes_at_node(self,
                                   upwind_node_ids_at_link,
                                   downwind_node_ids_at_link,
                                   mass_flux_at_link,
                                   suspended_sediment_mass_at_node_per_size,
                                   outlinks_at_node):

        outlinks_fluxes_at_node = np.zeros_like(self._zeros_at_node_for_fractions)
        inlinks_fluxes_at_node = np.zeros_like(self._zeros_at_node_for_fractions)
        shape = [np.size(self._active_links_ids), self._n_grain_sizes]
        total_outflux_at_node = np.zeros_like(self._zeros_at_node)
        total_influx_at_node = np.zeros_like(self._zeros_at_node)

        outlinks_fluxes_at_node, inlinks_fluxes_at_node, total_outflux_at_node, total_influx_at_node = cfuncs_ErosionDeposition.get_outin_fluxes(
            upwind_node_ids_at_link,
            downwind_node_ids_at_link,
            mass_flux_at_link,
            self._active_links_ids,
            outlinks_fluxes_at_node,
            inlinks_fluxes_at_node,
            total_outflux_at_node,
            total_influx_at_node,
            shape)

        indices_to_correct_flux = np.where(outlinks_fluxes_at_node > suspended_sediment_mass_at_node_per_size)
        if np.any(indices_to_correct_flux):
            ratios = np.divide(suspended_sediment_mass_at_node_per_size[indices_to_correct_flux],
                               outlinks_fluxes_at_node[indices_to_correct_flux])

            for i, (n, gs) in enumerate(zip(indices_to_correct_flux[0], indices_to_correct_flux[1])):
                out_links = self._grid.links_at_node[n, :][outlinks_at_node[n, :]]
                # Update the mass flux at link.
                mass_flux_at_link[out_links, gs] *= ratios[i, np.newaxis]

            outlinks_fluxes_at_node = np.zeros_like(self._zeros_at_node_for_fractions)
            inlinks_fluxes_at_node = np.zeros_like(self._zeros_at_node_for_fractions)
            total_outflux_at_node = np.zeros_like(self._zeros_at_node)
            total_influx_at_node = np.zeros_like(self._zeros_at_node)

            outlinks_fluxes_at_node, inlinks_fluxes_at_node, total_outflux_at_node, total_influx_at_node = cfuncs_ErosionDeposition.get_outin_fluxes(
                upwind_node_ids_at_link,
                downwind_node_ids_at_link,
                mass_flux_at_link,
                self._active_links_ids,
                outlinks_fluxes_at_node,
                inlinks_fluxes_at_node,
                total_outflux_at_node,
                total_influx_at_node,
                shape)

        self._grid.at_node['sediment__influx'][:] = total_influx_at_node
        self._grid.at_node['sediment__outflux'][:] = total_outflux_at_node

        return mass_flux_at_link, outlinks_fluxes_at_node, inlinks_fluxes_at_node

    def _calc_dzdt(self, size_class, dt=1):

        size_class = int(size_class)
        dt = dt
        xy_spacing = self._xy_spacing
        shape = self._grid_shape
        out = np.zeros_like(self._zeros_at_node)
        dzdt = cfuncs_ErosionDeposition.calc_flux_div_at_node(shape,
                                                              xy_spacing,
                                                              self._suspended__sediments_flux_at_link[
                                                                  :, size_class] * dt,
                                                              out)

        dzdt = -dzdt
        return dzdt

    def _calc_suspended_sediments_flux_at_link(self):

        ## Calc net suspended sediment flux
        size_class = np.where(np.any(self._suspended__sediments_flux_at_link, axis=0))[0].tolist()
        if len(size_class) > 0:
            if np.size(size_class) > 1:
                result = map(self._calc_dzdt,
                             size_class,
                             np.ones_like(size_class))
                self._suspended_dzdt_at_node_per_size[:, size_class] = np.asarray(list(result)).T

            else:
                dzdt = self._calc_dzdt(size_class=size_class[0])
                self._suspended_dzdt_at_node_per_size[:, size_class] = dzdt[:, np.newaxis]

    def _calc_DR(self):

        ## Pointers
        c_si_kg = self._c_kg
        surface_water__depth_at_node = self.grid.at_node['surface_water__depth']
        S = self._grid.at_node[self._slope]
        median_sizes = np.copy(self._grid.at_node['median_size__mass'])
        median_sizes[median_sizes == 0] = self._bedrock_median_size

        ## Get outflux water dischrage
        shape = (np.size(self._active_links_ids))
        q_at_node_raw = cfuncs_ErosionDeposition.sum_out_discharge(self._upwind_node_ids_at_link,
                                                                   np.abs(
                                                                       self._grid.at_link['surface_water__discharge']),
                                                                   self._active_links_ids,
                                                                   np.zeros_like(self._zeros_at_node),
                                                                   shape)

        # Exponential smoothing on discharge: depth (mass/continuity solution)
        # is smooth, but link discharge (momentum solution) shows high-frequency
        # chatter typical of explicit local-inertial overland-flow solvers. That
        # noise propagates straight into CQ, stream_power, and the calc_DR
        # branch condition. Smooth it here rather than chasing it downstream.
        if self._q_at_node_smoothed is None:
            self._q_at_node_smoothed = q_at_node_raw.copy()
        else:
            self._q_at_node_smoothed = (self._q_at_node_relaxation * q_at_node_raw
                                        + (1.0 - self._q_at_node_relaxation) * self._q_at_node_smoothed)
        q_at_node = self._q_at_node_smoothed

        ## Calc shear stress at node.
        max_downwind_gradient = self._grid.at_node['downwind__link_gradient']
        S[max_downwind_gradient <= 0] = 0
        self._tau_s = self._rho * self._g * S * surface_water__depth_at_node
        self._tau_s *= self._ft
        #
        # if self._veg_flag>0:
        #
        #
        #     veg_cover_at_node = self._grid.at_node['vegetation__cover_fraction']
        #     ## Option 1: Adjust tau_s based on vegetation
        #     ## Update roughness by cover
        #     # # section 3 in:       https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2004JF000249
        #     self._veg_roughness = self._veg_roughness_reference * (veg_cover_at_node / self._veg_cover_reference) ** self._omega_veg
        #     ft = (self._soil_roughness/ (self._veg_roughness + self._soil_roughness))**(3/2)
        #     self._tau_s *= ft
        #
        #     ## Option 2: Collins approach:
        #     # Adjust tau_c based on vegetation cover
        #     #self._tau_crit[:] = self._tau_crit_s + 50*veg_cover_at_node[:,np.newaxis]

        ## Flow width according to the fraction of the cell coevred by water
        # which is approximated by the relationshpip between surface runoff height and maxomum depression capacity (5.5 mm for shurb)
        # (Nunes et al., 2005, CATENA)
        flow_width = np.divide(surface_water__depth_at_node,
                               self._depression_depth)  # depression depth in meters
        flow_width[
            flow_width > self._grid.dx] = self._grid.dx  # greater than 1 means water depth is over the depression depth so flow width is the grid node width

        ## Calculation of  transport capacity at node
        TC = np.zeros_like(self._zeros_at_node_for_fractions)
        TC_eh = np.zeros_like(self._zeros_at_node_for_fractions)

        sg_c = self._SG
        rho_c = self._rho
        const_sg_g_rho = self._rho * (self._SG - 1) * self._g
        shape = (np.size(self._grid.core_nodes), np.shape(self._zeros_at_node_for_fractions)[1])
        # calc_TC requires a strictly 2D (n_nodes, n_sizes) buffer here, but
        # with a single grain size Landlab squeezes this field down to 1D
        # (n_nodes,) - reshape(-1, 1) restores the expected shape as a view
        # (no copy), a no-op for the normal already-2D multi-size case.
        grains_classes_size = self._grid.at_node['grains_classes__size']
        if np.ndim(grains_classes_size) == 1:
            grains_classes_size = grains_classes_size.reshape(-1, 1)
        self._TC = cfuncs_ErosionDeposition.calc_TC(
            self._alpha,
            self._beta,
            median_sizes,
            grains_classes_size,
            self._tau_s,
            TC,
            sg_c,
            rho_c,
            self._grid.core_nodes,
            const_sg_g_rho,
            shape)

        # Smoothly choke off transport capacity as the current sediment/water
        # volume ratio (self._sum_c_si, from the previous concentration update)
        # approaches the physical packing limit Cv_max, instead of an abrupt
        # "dump everything" cutoff at a somewhat arbitrary ratio of 1.
        Cv_taper = 1.0 - (self._sum_c_si / self._Cv_max) ** self._Cv_taper_exponent
        np.clip(Cv_taper, 0.0, 1.0, out=Cv_taper)
        self._TC = self._TC * Cv_taper[:, np.newaxis]

        # a = cfuncs_ErosionDeposition.calc_TC_EH_with_discharge(
        #     self._grid.at_node['grains_classes__size'],
        #     self._tau_s,
        #     q_at_node,
        #     surface_water__depth_at_node,
        #     TC_eh,
        #     sg_c,
        #     rho_c,
        #     self._grid.core_nodes,
        #     const_sg_g_rho,
        #     shape)
        #
        # print('TC Yalin', np.median(self._TC),' TC EU', np.median(a))

        ## Calculation of Dc (detachment rate)
        if self._detachment_model == 'shear':
            self._Dc = cfuncs_ErosionDeposition.calc_Dc(self._tau_s,
                                                        self._tau_crit,
                                                        self._grid.core_nodes,
                                                        np.zeros_like(self._zeros_at_node_for_fractions),
                                                        self._kr,
                                                        shape)

        elif self._detachment_model == 'stream_power':
            # RHEM V2.3 (Al-Hamdan et al., 2012b): Dc = K_omega * omega,
            # omega = rho * g * S * q  (stream power, kg/s^3). No critical-shear
            # threshold -- detachment starts as soon as concentrated flow starts.
            # `ft` (vegetation/soil shear-partitioning factor, same one applied
            # to tau_s above) is folded in here too, so a fixed "bare, just-burned"
            # k_omega baseline still gets the time-varying vegetation-recovery
            # reduction from `ft` -- otherwise it would never reach this path.
            stream_power = self._rho * self._g * S * q_at_node * self._ft
            self._Dc = cfuncs_ErosionDeposition.calc_Dc_stream_power(stream_power,
                                                                     self._grid.core_nodes,
                                                                     np.zeros_like(self._zeros_at_node_for_fractions),
                                                                     self._k_omega,
                                                                     shape)

        else:
            raise ValueError(
                "detachment_model must be 'shear' or 'stream_power', got %r" % self._detachment_model)

        ## Get the total mass flux
        CQ = cfuncs_ErosionDeposition.calc_CQ(
            c_si_kg,
            np.zeros_like(self._zeros_at_node_for_fractions),
            q_at_node,
            self._grid.core_nodes,
            shape,
            int(self._grid.dx))

        ## Calcuation of net erosion/deposition at node
        DR_new = cfuncs_ErosionDeposition.calc_DR(flow_width,
                                                  CQ,
                                                  self._TC,
                                                  self._Dc,
                                                  self._vs[0, :],
                                                  self._grid.core_nodes,
                                                  q_at_node,
                                                  np.zeros_like(CQ),
                                                  self._grid.dx,
                                                  shape)

        # Under-relaxation: blend with last step's DR to damp the
        # detachment/deposition bang-bang oscillation (TC/CQ flipping which
        # branch is active every step). DR_relaxation=1 reproduces the
        # original, undamped behavior exactly; lower values trade responsiveness
        # for smoothness. self._DR here is still last step's value at this point.
        self._DR[:] = self._DR_relaxation * DR_new + (1.0 - self._DR_relaxation) * self._DR
        # self._yuval = np.divide(CQ,
        #                          flow_width[:,np.newaxis]*self._TC,where=self._TC>0)

    def _calc_load_flux(self):

        ## Pointers
        water_surface_grad_at_link = self._grid.calc_grad_at_link('water_surface__elevation')
        suspended__sediments_concentration_at_link = self._suspended__sediments_concentration_at_link
        mass_flux_at_link = self._mass_flux_at_link
        suspended_sediment_mass_at_node_per_size = self._suspended_sediment_mass_at_node_per_size
        upwind_node_ids_at_link = self._upwind_node_ids_at_link
        downwind_node_ids_at_link = self._downwind_node_ids_at_link
        surface_water__depth_at_node = self.grid.at_node['surface_water__depth']
        surface_water__depth_at_node[surface_water__depth_at_node < 10 ** -8] = 10 ** -8
        surface_water__depth_at_node_expand = np.expand_dims(surface_water__depth_at_node, -1)
        q_water_at_link = self.grid.at_link[
            'surface_water__discharge']  # surface water discharge units are m**3/s -> From OverlandFlow component
        q_water_at_link[self._inactive_links] = 0

        ## mass concentration at link
        suspended__sediments_concentration_at_link[:] = np.divide(suspended_sediment_mass_at_node_per_size[
                                                                      upwind_node_ids_at_link, :],
                                                                  surface_water__depth_at_node_expand[
                                                                      upwind_node_ids_at_link,
                                                                      :] * self._grid.dx * self._grid.dx)
        ## Find the outlinks for each node.
        outlinks_at_node = self.grid.link_at_node_is_downwind(water_surface_grad_at_link)

        ## Calc mass flux at link
        shape = [np.size(self._active_links_ids), self._n_grain_sizes]
        mass_flux_at_link[:] = cfuncs_ErosionDeposition.calc_flux_at_link_per_size(q_water_at_link,
                                                                                     suspended__sediments_concentration_at_link,
                                                                                     self._grid.active_links,
                                                                                     np.zeros_like(mass_flux_at_link),
                                                                                     shape)
        mass_flux_at_link = np.abs(mass_flux_at_link)

        ## Get the in/out suspended mass fluxe at NODE
        mass_flux_at_link[:], outlinks_fluxes_at_node, inlinks_fluxes_at_node = self._calc_inout_fluxes_at_node(
            upwind_node_ids_at_link,
            downwind_node_ids_at_link,
            mass_flux_at_link,
            suspended_sediment_mass_at_node_per_size,
            outlinks_at_node)

        self._outlinks_fluxes_at_node[:] = outlinks_fluxes_at_node
        self._inlinks_fluxes_at_node[:] = inlinks_fluxes_at_node

        ## Get the suspended mass flux at LINK
        shape = np.shape(self._suspended__sediments_flux_at_link)
        self._suspended__sediments_flux_at_link[:] = cfuncs_ErosionDeposition.calc_flux_at_link(self._grid.dx,
                                                                                                self._sigma,
                                                                                                self._phi,
                                                                                                np.abs(
                                                                                                    mass_flux_at_link),
                                                                                                -np.sign(
                                                                                                    water_surface_grad_at_link),
                                                                                                self._suspended__sediments_flux_at_link,
                                                                                                shape)

        ## Calculate sediment-load flux at link
        self._calc_suspended_sediments_flux_at_link()

    def _map_upwind_downwind_nodes_to_links(self,
                                            elev_field='water_surface__elevation'):

        ## Map upwind/downwind node id to links
        upwind_node_ids_at_link = self._grid.map_value_at_max_node_to_link(elev_field,
                                                                           self._nodes_flatten).astype('int')
        self._upwind_node_ids_at_link = upwind_node_ids_at_link

        downwind_node_ids_at_link = self._grid.map_value_at_min_node_to_link(elev_field,
                                                                             self._nodes_flatten).astype('int')
        self._downwind_node_ids_at_link = downwind_node_ids_at_link

    def _calc_erosion_deposition(self, ):

        # Pointers
        surface_water__depth_at_node = self.grid.at_node['surface_water__depth']
        surface_water__depth_at_node[surface_water__depth_at_node < 10 ** -8] = 10 ** -8
        surface_water__depth_at_node_expand = np.expand_dims(surface_water__depth_at_node, -1)
        soil_depth = self._grid.at_node['soil__depth']
        # With a single grain size, grains__mass loses its second axis
        # (n_nodes,) instead of (n_nodes, n_sizes). reshape(-1, 1) restores
        # a 2D view (not a copy) so in-place writes through this local name
        # still propagate back into the real grid field, and downstream
        # axis=1 operations below don't blow up - a no-op for the normal
        # already-2D multi-size case.
        grain_mass_at_node = self.grid.at_node['grains__mass']
        if np.ndim(grain_mass_at_node) == 1:
            grain_mass_at_node = grain_mass_at_node.reshape(-1, 1)
        q_water_at_link = self.grid.at_link[
            'surface_water__discharge']  # surface water discharge units are m**3/s -> From OverlandFlow component
        q_water_at_link[self._inactive_links] = 0
        deposited_suspended_sediments_masss_at_node = self._deposited_suspended_sediments_masss_at_node
        suspended_sediment_mass_at_node_per_size = self._suspended_sediment_mass_at_node_per_size
        deposited_suspended_sediments_dz_at_node = self._deposited_suspended_sediments_dz_at_node
        local_sediment_mass_flux_at_node_per_size = self._local_sediment_mass_flux_at_node_per_size  # soil and bedrock entrainment
        detached_bedrock_rate_dz = self._detached_bedrock_rate_dz  # to suspended
        detached_soil_rate_dz = self._detached_soil_rate_dz  # to suspended
        detached_bedrock_mass = self._detached_bedrock_mass  # to suspended
        detached_soil_mass = self._detached_soil_mass  # to suspended

        c_si = self._c_si
        c_kg = self._c_kg
        grain_fractions_at_node = self._grain_fractions_at_node
        suspended_fraction_at_node = self._suspended_fraction_at_node
        grain_mass_at_node[grain_mass_at_node < 10 ** -10] = 10 ** -10

        ## Calc soil exponent
        soil_e_expo = (1 - (np.exp(
            (-soil_depth) / self._cover_depth_star))
                       )

        ## Suspended load total volume
        temp_suspended_sediment_mass_at_node_per_size = np.copy(suspended_sediment_mass_at_node_per_size)
        temp_suspended_sediment_mass_at_node_per_size[temp_suspended_sediment_mass_at_node_per_size < 0] = 0
        temp_suspended_sediment_volume_at_node_per_size = temp_suspended_sediment_mass_at_node_per_size / self._sigma  # remmeber, here, no porosity correction
        temp_suspended_sediment_volume_at_node_per_size[temp_suspended_sediment_volume_at_node_per_size < 0] = 0

        ## Concentrations (volumetric and mass)
        c_si[:] = np.divide(temp_suspended_sediment_volume_at_node_per_size,
                            surface_water__depth_at_node_expand * self.grid.dx ** 2,
                            out=np.ones_like(temp_suspended_sediment_volume_at_node_per_size))

        # mass concentration
        c_kg[:] = np.divide(temp_suspended_sediment_mass_at_node_per_size,
                            surface_water__depth_at_node_expand * self.grid.dx ** 2,
                            out=np.ones_like(temp_suspended_sediment_volume_at_node_per_size))

        # Calc mass fraction of all size classes
        temp_suspended_sediment_mass_at_node_per_size[
            temp_suspended_sediment_mass_at_node_per_size < 10 ** -10] = 10 ** -10
        suspended_fraction_at_node[:] = cfuncs_ErosionDeposition.calc_concentration(
            np.ones_like(temp_suspended_sediment_mass_at_node_per_size),
            temp_suspended_sediment_mass_at_node_per_size,
            np.sum(temp_suspended_sediment_mass_at_node_per_size, 1),
            np.shape(temp_suspended_sediment_mass_at_node_per_size)
        )

        # Total sediment/water volume ratio at node. Used to smoothly choke off
        # transport capacity as concentration approaches the physical packing
        # limit (see self._Cv_max), instead of an abrupt full-dump cutoff.
        sum_c_si = cfuncs_ErosionDeposition.grain_size_sum_at_node(c_si,
                                                                   np.zeros_like(self._zeros_at_node),
                                                                   np.shape(c_si))
        self._sum_c_si = sum_c_si

        ## Calculation of erosion/deposition at node
        self._calc_DR()  # DR returns in units of kg/(m^2*s)

        ## Calc net change for both layer
        core_nodes = self._grid.core_nodes
        factor_convert_mass_to_dz_c = self._sigma * (1 - self._phi) * self._grid.dx ** 2
        factor_convert_mass_to_dz_bedrock_c = (1 - self._phi) * self._sigma * self._grid.dx ** 2
        shape = (np.size(core_nodes), np.shape(self._zeros_at_node_for_fractions)[1])
        dx_c = self._grid.dx

        ## Calc mass concentration at node
        grain_fractions_at_node[:] = cfuncs_ErosionDeposition.calc_concentration(np.ones_like(grain_mass_at_node),
                                                                                 grain_mass_at_node,
                                                                                 np.sum(grain_mass_at_node, axis=1),
                                                                                 np.shape(grain_mass_at_node)
                                                                                 )
        ## Calc Erosion/Deposition at node
        (detached_soil_mass[:],
         detached_bedrock_mass[:],
         detached_soil_rate_dz[:],
         detached_bedrock_rate_dz[:],
         deposited_suspended_sediments_masss_at_node[:],
         deposited_suspended_sediments_dz_at_node[:]) = cfuncs_ErosionDeposition.calc_detached_deposited(self._DR,
                                                                                                         np.abs(
                                                                                                             self._DR),
                                                                                                         grain_mass_at_node,
                                                                                                         grain_fractions_at_node,
                                                                                                         detached_soil_mass,
                                                                                                         detached_bedrock_mass,
                                                                                                         suspended_fraction_at_node,
                                                                                                         self._bedrock_grain_fractions,
                                                                                                         temp_suspended_sediment_mass_at_node_per_size,
                                                                                                         deposited_suspended_sediments_masss_at_node,
                                                                                                         deposited_suspended_sediments_dz_at_node,
                                                                                                         detached_soil_rate_dz,
                                                                                                         detached_bedrock_rate_dz,
                                                                                                         soil_e_expo,
                                                                                                         core_nodes,
                                                                                                         factor_convert_mass_to_dz_c,
                                                                                                         factor_convert_mass_to_dz_bedrock_c,
                                                                                                         shape,
                                                                                                         dx_c
                                                                                                         )
        # print(np.max(deposited_suspended_sediments_dz_at_node))
        # ## Update mass fluxes (deatched/deposited) at node
        # local_sediment_mass_flux_at_node_per_size[
        # :] = detached_bedrock_mass + detached_soil_mass  # detached flux to suspended

    def _calc_stable_dt(self):

        max_downwind_gradient = self._grid.at_node['downwind__link_gradient']
        max_upwind_gradient = self._grid.at_node['upwind__link_gradient']
        S = self._grid.at_node[self._slope]
        detached_bedrock_rate_dz = self._detached_bedrock_rate_dz  # to suspended
        detached_soil_rate_dz = self._detached_soil_rate_dz  # to suspended
        deposited_suspended_sediments_dz_at_node = self._deposited_suspended_sediments_dz_at_node
        deposited_suspended_sediments_masss_at_node = self._deposited_suspended_sediments_masss_at_node
        temp_suspended_sediment_mass_at_node_per_size = np.copy(self._suspended_sediment_mass_at_node_per_size)
        temp_suspended_sediment_mass_at_node_per_size[temp_suspended_sediment_mass_at_node_per_size < 0] = 0
        suspended_sediment_mass_at_node_per_size = self._suspended_sediment_mass_at_node_per_size
        outlinks_fluxes_at_node = self._outlinks_fluxes_at_node

        # Condition 1:
        # Stable incision dz in each node will be half of the maximal DONWIND gradient
        # A minimal elevation diffrence threshold for incision is set. Below this value,
        # incision assumed to be zero (slope is VERY low).
        stable_incision_dz = (max_downwind_gradient * self._grid.dx) / 2  # topographic slope
        S[stable_incision_dz <= 0] = 0
        stable_incision_dz[
            stable_incision_dz <= 0] = np.inf  # set to infinity because slope is set to zero for this node (no erosion).
        E_tot = (detached_bedrock_rate_dz + detached_soil_rate_dz) - deposited_suspended_sediments_dz_at_node
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
        # Make sure deposited mass is not greater than what exist in the flow
        if np.any(
                deposited_suspended_sediments_masss_at_node > 10 ** -10):  # some error that I allow for not get into small time steps all the time.
            self._stable_deposition_rate = \
                np.max([np.min(
                    np.divide(temp_suspended_sediment_mass_at_node_per_size,
                              deposited_suspended_sediments_masss_at_node,
                              where=deposited_suspended_sediments_masss_at_node > 10 ** -10,
                              out=np.ones_like(deposited_suspended_sediments_masss_at_node) * np.inf)), 1])
        else:
            self._stable_deposition_rate = np.inf
        self._stable_deposition_rate = np.inf

        # Condition 3.
        # Calc stable DEPOSITION depth:
        # Stable deposition dz in each node will be half of the maximal UPWIND gradient
        stable_deposition_dz = (np.abs(max_upwind_gradient) *
                                self._grid.dx) / 2  # Elevation diffrence of node to its UPWIND node
        stable_deposition_dz[
            stable_deposition_dz <= self._max_flipped_deposition_slope
            ] = self._max_flipped_deposition_slope * self._grid.dx

        net_deposition_dz = deposited_suspended_sediments_dz_at_node - (
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
        sum__suspended_mass_flux_at_node_per_size = self._suspended_dzdt_at_node_per_size * self._grid.dx ** 2 * self._sigma * (
                1 - self._phi)
        sum__suspended_mass_flux_at_node = cfuncs_ErosionDeposition.grain_size_sum_at_node(
            sum__suspended_mass_flux_at_node_per_size,
            np.zeros_like(self._zeros_at_node),
            np.shape(sum__suspended_mass_flux_at_node_per_size))

        if np.any(sum__suspended_mass_flux_at_node < -self._min_suspended_mass_to_del):
            nodes_with_removed_suspended_mass = \
                np.where(sum__suspended_mass_flux_at_node < -self._min_suspended_mass_to_del)[0]

            suspended_masss_at_node_per_size = suspended_sediment_mass_at_node_per_size[
                nodes_with_removed_suspended_mass, :]
            if np.ndim(outlinks_fluxes_at_node) > 2:
                outfluxes_masss_at_node_per_size = np.sum(np.abs(outlinks_fluxes_at_node), axis=1)[
                    nodes_with_removed_suspended_mass, :]
            else:
                outfluxes_masss_at_node_per_size = np.abs(outlinks_fluxes_at_node)[
                    nodes_with_removed_suspended_mass, :]

            self._stable_dt_mass = np.divide(suspended_masss_at_node_per_size,
                                             outfluxes_masss_at_node_per_size,
                                             out=np.ones_like(outfluxes_masss_at_node_per_size) * np.inf,
                                             where=outfluxes_masss_at_node_per_size > self._min_suspended_mass_to_del)

            self._stable_dt_mass = np.min(self._stable_dt_mass)
            if self._stable_dt_mass == 0:
                self._stable_dt_mass = np.inf
        else:
            self._stable_dt_mass = np.inf

        # self._stable_deposition_rate = np.inf  ## check if this is nesseary
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
        # Same field-squeeze issue as _calc_erosion_deposition above: with a
        # single grain size, grains__mass loses its second axis. Restore
        # a 2D view (not a copy, so the in-place += below still writes
        # through to the real field) - a no-op for the normal multi-size
        # case, which is already 2D.
        grain_masss = self.grid.at_node['grains__mass']  # kg/m2
        if np.ndim(grain_masss) == 1:
            grain_masss = grain_masss.reshape(-1, 1)
        soil_depth = self._grid.at_node['soil__depth']
        bedrock = self._grid.at_node['bedrock__elevation']
        topo = self._grid.at_node['topographic__elevation']
        suspended_sediment_mass_at_node = self.grid.at_node['suspended__sediments_masss']
        suspended_sediment_mass_at_node_per_size = self._suspended_sediment_mass_at_node_per_size
        detached_soil_mass = self._detached_soil_mass
        detached_bedrock_mass = self._detached_bedrock_mass

        # Load fluxes per time step
        net_suspended_mass_flux_from_srrnds = self._suspended_dzdt_at_node_per_size * self._grid.dx ** 2 * self._sigma * dt * (
                1 - self._phi)  # NET flux after div. of suspended sediment at node

        ## Deposited from load
        deposited_suspended_sediments_masss_at_node = self._deposited_suspended_sediments_masss_at_node[
                                                            :] * dt  # Deposited mass of suspended sediment at the node
        # deposited_suspended_sediments_masss_at_node[:]= np.min((deposited_suspended_sediments_masss_at_node, suspended_sediment_mass_at_node_per_size+net_suspended_mass_flux_from_srrnds), 0)
        deposited_suspended_sediments_masss_at_node[:] = np.min(
            (deposited_suspended_sediments_masss_at_node, suspended_sediment_mass_at_node_per_size), 0)

        ## Detached to load
        ## Update mass fluxes (deatched/deposited) at node
        local_sediment_mass_flux_at_node_per_size = (
                                                                  detached_soil_mass + detached_bedrock_mass) * dt  # Enrichment mass of suspended sediment at the node

        # Update load
        suspended_sediment_mass_at_node_per_size[:] = suspended_sediment_mass_at_node_per_size + (
                    local_sediment_mass_flux_at_node_per_size + net_suspended_mass_flux_from_srrnds)
        suspended_sediment_mass_at_node_per_size[
            :] = suspended_sediment_mass_at_node_per_size - deposited_suspended_sediments_masss_at_node
        suspended_sediment_mass_at_node_per_size[suspended_sediment_mass_at_node_per_size < 0] = 0
        suspended_sediment_mass_at_node[:] = np.sum(suspended_sediment_mass_at_node_per_size, axis=1)

        # dz change in bedrock/soil layers
        detached_bedrock_rate_dz = self._detached_bedrock_rate_dz * dt
        detached_soil_mass = self._detached_soil_mass * dt
        detached_bedrock_mass = self._detached_bedrock_mass * dt
        dmass = deposited_suspended_sediments_masss_at_node[:] - detached_soil_mass[:]
        self._yuval = dmass
        if self._change_topo_flag:
            grain_masss[:] += dmass / (self.grid.dx ** 2)
            grain_masss[grain_masss < 0] = 0

            soil_depth[:] = (np.sum(grain_masss, axis=1) / (self._sigma)) / (1 - self._phi)
            bedrock[:] -= detached_bedrock_rate_dz[:]
            topo[:] = soil_depth[:] + bedrock[:]

        if self._veg_flag > 0:
            ## Redcue veg cover by flow action
            veg_cover_fraction_at_cell = self._grid.at_cell['vegetation__cover_fraction']
            if self._veg_flag == 1:

                # Option 1: Based on previous Collins and Erkan papers
                # Same squeeze-to-1D issue as in _calc_DR above: with a
                # single grain size, grains_classes__size loses its second
                # axis, so the [core_nodes, :] 2D indexing below fails.
                grains_classes_size = self._grid.at_node['grains_classes__size']
                if np.ndim(grains_classes_size) == 1:
                    grains_classes_size = grains_classes_size.reshape(-1, 1)
                median_size_index = np.where(
                    grains_classes_size[self._grid.core_nodes, :] == self._grid.at_node['median_size__mass'][
                        self._grid.core_nodes, np.newaxis])[1]
                self._grid.at_node['excess___stress'][self._grid.core_nodes] = self._tau_s[self._grid.core_nodes] - \
                                                                               self._tau_crit[
                                                                                   self._grid.core_nodes, median_size_index]

                excess_stress_at_cell = self._grid.map_node_to_cell('excess___stress')
                above_tau_c = np.where(excess_stress_at_cell > 0)
                dv_dt = np.zeros_like(veg_cover_fraction_at_cell)
                dv_dt[above_tau_c] = self._Kv * veg_cover_fraction_at_cell[above_tau_c] * (
                excess_stress_at_cell[above_tau_c])
                dv = dv_dt * dt
                dv[dv >= veg_cover_fraction_at_cell] = veg_cover_fraction_at_cell[dv >= veg_cover_fraction_at_cell] - (
                        10 ** -3)
                dv_change_ratio = np.divide(dv,
                                            veg_cover_fraction_at_cell[:],
                                            where=veg_cover_fraction_at_cell > 10 ** -3,
                                            out=np.zeros_like(dv))
                dv_change_ratio_inverse = 1 - dv_change_ratio
                dv_change_ratio_inverse[dv_change_ratio_inverse < 0.01] = 0.01

            else:
                # Option 2: based on erosion and reference length (root depth)
                # dz_abs = np.abs(np.sum(dmass, 1) / (self._sigma * self.grid.dx ** 2) / (1 - self._phi))
                dmass = deposited_suspended_sediments_masss_at_node[:] - (
                        detached_soil_mass[:] + detached_bedrock_mass[:])

                sum_dmass = np.sum(dmass, 1)
                # sum_dmass[sum_dmass>0] = 0

                dz_abs = np.abs(sum_dmass) / (self._sigma * self.grid.dx ** 2) / (1 - self._phi)
                dz_net_at_cell = self._grid.map_node_to_cell(dz_abs)
                root_depth = 0.1

                dv = (dz_net_at_cell / root_depth) * veg_cover_fraction_at_cell * 10
                dv[dv >= veg_cover_fraction_at_cell] = veg_cover_fraction_at_cell[dv >= veg_cover_fraction_at_cell] - (
                            10 ** -3)
                dv_change_ratio = np.divide(dv,
                                            veg_cover_fraction_at_cell[:],
                                            where=veg_cover_fraction_at_cell > 10 ** -3,
                                            out=np.zeros_like(dv))
                dv_change_ratio_inverse = 1 - dv_change_ratio
                # dv_change_ratio_inverse[dv_change_ratio_inverse<0.01] = 0.01
                # dv_change_ratio_inverse = np.exp(-(dz_net_at_cell/root_depth))
                # print(np.min(dv_change_ratio_inverse))

            # dv_change_ratio_inverse = [False, False]
            if np.any(dv_change_ratio_inverse):
                self._grid.at_cell['vegetation__live_biomass'][
                    :] *= dv_change_ratio_inverse  ##change ":" to "above_tau_c" in case of option #1
                self._grid.at_cell['vegetation__dead_biomass'][:] *= dv_change_ratio_inverse
                self._grid.at_cell['vegetation__live_leaf_area_index'][:] *= dv_change_ratio_inverse
                self._grid.at_cell['vegetation__dead_leaf_area_index'][:] *= dv_change_ratio_inverse

                veg_cover_fraction_at_cell[:] *= dv_change_ratio_inverse
                veg_cover_fraction_at_cell[veg_cover_fraction_at_cell <= 0] = 10 ** -8

            ## Map veg cover from cell back to the node
            veg_cover_at_node = self._grid.at_node['vegetation__cover_fraction']
            veg_cover_at_node[self._nodes_at_cell] = veg_cover_fraction_at_cell[:]

            ## Update roughness by cover
            # Calc roughness at node, taking into account vegetation cover roughness
            self._veg_roughness = self._veg_roughness_reference * (
                        veg_cover_at_node / self._veg_cover_reference) ** self._omega_veg
            bare_soil_fraction = 1 - veg_cover_at_node
            self._grid.at_node['mannings_n'][:] = (
                                                              bare_soil_fraction * self._soil_roughness) + veg_cover_at_node * self._veg_roughness

            ## Map the roughness from node to link
            self._grid.at_link['mannings_n'][:] = self._grid.at_node['mannings_n'][self._upwind_node_ids_at_link]

    def update_vegetated_roughness(self):

        bare_soil_fraction = 1 - self._grid.at_node['vegetation_cover__fraction']
        self._veg_roughness = self._veg_roughness_reference * self._grid.at_node[
            'vegetation_cover__fraction'] ** self._omega_veg

        self._grid.at_node['mannings_n'][:] = (bare_soil_fraction *
                                               self._soil_roughness) + (self._grid.at_node[
                                                                            'vegetation_cover__fraction'] * self._veg_roughness)

        ## Map the roughness from node to link
        self._grid.at_link['mannings_n'][:] = self._grid.at_node['mannings_n'][self._upwind_node_ids_at_link]

        self._ft = (self._soil_roughness / (self._veg_roughness + self._soil_roughness)) ** (3 / 2)

