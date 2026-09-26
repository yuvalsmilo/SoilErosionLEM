"""Overland Flow Erosion and Deposition Component.
Author: Yuval Shmilovitz
September 2026
"""

import numpy as np
from landlab import Component
import cfuncs_ErosionDeposition

_TEN_MINUS_THREE = 1e-3
_NEGLIG = 10**-8

class OverlandflowErosionDeposition(Component):
    """Simulate erosion and deposition by overland flow.

    This component models sediment transport processes driven by surface water flow,
    including detachment from bedrock and soil, total load sediment transport, and
    deposition. It uses a transport capacity approach with multiple grain sizes.
    Vegetation-erosion feedback dynamics are optional.

    References
    ----------
    Al-Hamdan, O. Z., Pierson, F. B., Nearing, M. A., Williams, C. J., Stone,
        J. J., Kormos, P. R., Boll, J., & Weltz, M. A. (2012). Concentrated
        flow erodibility for physically based erosion models: Temporal
        variability in disturbed and undisturbed rangelands. Water Resources
        Research, 48, W07504.
    Foster, G. R. (1982). Modeling the erosion process. In C. T. Haan, H. P.
        Johnson, & D. L. Brakensiek (Eds.), Hydrologic Modeling of Small
        Watersheds (pp. 297-380). ASAE Monograph No. 5, American Society of
        Agricultural Engineers.
    Istanbulluoglu, E., & Bras, R. L. (2005). Vegetation-modulated landscape
        evolution: Effects of vegetation on landscape processes, drainage
        density, and topography. Journal of Geophysical Research, 110(F2).
    Komar, P. D. (1987). Selective grain entrainment by a current from a bed of
        mixed sizes: A reanalysis. Journal of Sedimentary Research, 57(2), 203-211.
    Nunes, J. P., Vieira, G. N., Seixas, J., Goncalves, P., & Carvalhais, N.
        (2005). Evaluating the MEFIDIS model for runoff and soil erosion
        prediction during rainfall events. Catena, 61(2-3), 210-228.
    Yalin, M. S. (1963). An expression for bed-load transportation. Journal of
        the Hydraulics Division, ASCE, 89(3), 221-250.
    """

    @staticmethod
    def _ensure_2d(field):
        """Return a per-node grain-size field as a 2D (n_nodes, n_sizes) array.
        """
        if field.ndim == 1:
            return field.reshape(-1, 1)
        return field

    _name = "OverlandflowErosionDeposition"
    _unit_agnostic = True

    _info = {
        "surface_water__depth": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Surface water depth",
        },
        "topographic__elevation": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Land surface topographic elevation",
        },
        "total_load__sediments_mass": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "kg",
            "mapping": "node",
            "doc": "mass of total_load sediment per grain size at the node",
        },
        "sediment__influx": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m3/s",
            "mapping": "node",
            "doc": "Sediment flux entering each node (volume per unit time)",
        },
        "sediment__outflux": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m3/s",
            "mapping": "node",
            "doc": "Sediment flux leaving each node (volume per unit time)",
        },
        "excess___stress": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "Pa",
            "mapping": "node",
            "doc": "Excess shear stress above critical threshold",
        },
        "grains__mass": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "kg/m^2",
            "mapping": "node",
            "doc": "Sediment mass per size class",
        },
    }

    def __init__(
        self,
        grid,
        # Physical constants
        fluid_density=1000.0,
        g=9.81,
        sigma=2650.0,
        phi=0.4,
        # Erosion/transport parameters
        tau_crit=1.165,
        kr=0.0001,
        cover_depth_star=0.1,
        # Detachment model selection
        detachment_model='shear',  # 'shear' (Dc = kr*(tau_s - tau_crit)) or 'stream_power' (Dc = k_omega * rho*g*S*q, RHEM V2.3)
        k_omega=7.74e-3,  # stream-power erodibility coefficient [s^2/m^2], used when detachment_model='stream_power'
        q_at_node_relaxation=1,  # exponential smoothing on q_at_node (1=no smoothing/raw discharge, lower=more smoothing of the momentum-solution chatter)
        ft=1,  # vegetation shear-partitioning factor; 1 = no reduction
        dr_relaxation=0.5,  # under-relaxation on net erosion/deposition rate (1=no damping, lower=more damping of the detachment/deposition)
        cv_max=0.9,  # sediment/water volume ratio at which transport capacity is fully choked off
        cv_taper_exponent=2,  # shape of the capacity taper as concentration approaches cv_max
        # Grain size parameters
        submerged_specific_gravity=1.65,
        bedrock_grain_sizes=None,
        # Settling velocity parameters
        C1=18,
        C2=0.4,
        kinematic_viscosity=9.2e-7,
        # Transport hidden coefficients
        alpha=0.045,
        beta=-0.68,
        # Flow geometry
        depression_depth=0.0055,
        max_deposition_slope=0.01,
        # Vegetation parameters
        veg_flag=0,
        vegetation_coefficient=3e-8,
        vegetation_exponent=0.5,
        veg_roughness_reference=0.6,
        veg_cover_reference=0.8,
        soil_roughness=0.025,
        root_depth=0.1,
        # Model options
        slope='water_surface__slope',
        change_topo_flag=True,
        max_stable_dt=60,
    ):
        """Initialize the OverlandflowErosionDeposition component.

        Parameters
        ----------
        grid : ModelGrid
            A Landlab grid object
        fluid_density : float, optional
            Density of the fluid [kg/m³]. Default: 1000.0
        g : float, optional
            Gravitational acceleration [m/s²]. Default: 9.81
        sigma : float, optional
            Density of sediment grains [kg/m³]. Default: 2650.0
        phi : float, optional
            Soil porosity [-]. Default: 0.4
        tau_crit : float, optional
            Critical shear stress for erosion [Pa]. Default: 1.165
        kr : float, optional
            Erosion detachment rate coefficient [kg/(Pa·s·m²)]. Default: 0.0002
        cover_depth_star : float, optional
            Characteristic depth for soil cover effect [m]. Default: 0.1
        detachment_model : str, optional
            'shear' (Dc = kr*(tau_s - tau_crit)) or 'stream_power'
            (Dc = k_omega * rho*g*S*q, RHEM V2.3). Default: 'shear'
        k_omega : float, optional
            Stream-power erodibility coefficient [s^2/m^2], used when
            detachment_model='stream_power'. Default: 7.74e-3
        q_at_node_relaxation : float, optional
            Exponential smoothing factor on discharge at node, in (0, 1].
            1 = no smoothing, lower = more smoothing of the
            momentum-solution chatter. Default: 1
        ft : float, optional
            Shear-partitioning factor folded into the stream_power
            detachment calculation. Default: 1 (no reduction)
        dr_relaxation : float, optional
            Under-relaxation factor on the net erosion/deposition rate, in
            (0, 1]. 1 = no damping (raw rate used every step), lower = more
            damping of the detachment/deposition oscillation
            (TC/CQ flipping which branch is active every step). Default: 0.5
        cv_max : float, optional
            Sediment/water volume ratio at which transport capacity is fully
            choked off)
        cv_taper_exponent : float, optional
            Shape of the transport-capacity taper as concentration
            approaches cv_max. Default: 2
        submerged_specific_gravity : float, optional
            Submerged specific gravity of sediment [-]. Default: 1.65
        bedrock_grain_sizes : array-like, optional
            Grain size distribution of bedrock. Default: None (uses grid values)
        C1, C2 : float, optional
            Settling velocity equation coefficients. Default: 18, 0.4
        kinematic_viscosity : float, optional
            Kinematic viscosity of water [m²/s]. Default: 9.2e-7
        alpha : float, optional
            Critical dimensionless shear stress for the median size Default: 0.045
        beta : float, optional
            Empirical constant for dimensionless shear stress. Default: -0.68
        depression_depth : float, optional
            Characteristic surface depression depth [m]. Default: 0.0055
        max_deposition_slope : float, optional
            Maximum inverse slope for deposition [m/m]. Default: 0.01
        veg_flag : int, optional
            Vegetation-erosion dynamics flag (0=off, >0=erosion-depth-based
            removal via _update_vegetation_by_erosion). Default: 0
        vegetation_coefficient : float, optional
            Vegetation removal coefficient. Default: 3e-8
        vegetation_exponent : float, optional
            Vegetation-erosion dynamics exponent. Default: 0.5
        veg_roughness_reference : float, optional
            Reference Manning's n for vegetation. Default: 0.6
        veg_cover_reference : float, optional
            Reference vegetation cover fraction. Default: 0.8
        soil_roughness : float, optional
            Manning's n for bare soil. Default: 0.025
        root_depth : float, optional
            Root depth [m]. Default: 0.1
        slope : str, optional
            Name of slope field in grid. Default: "water_surface__slope"
        change_topo_flag : bool, optional
            Whether to update topography based on erosion/deposition. Default: True
        max_stable_dt : float, optional
            Maximum stable time step. Default: 60
        """
        super().__init__(grid)

        # Verify required slope field exists
        assert slope in grid.at_node, f"Slope field '{slope}' not found in grid"

        # Initialize output fields
        self.initialize_output_fields()

        # Grid geometry
        self._grid_shape = np.array(self.grid.shape)
        self._xy_spacing = np.array(self.grid.spacing)
        self._dx = self.grid.dx
        self._dx_squared = self._dx ** 2  # Cache frequently used value
        self._num_nodes = np.prod(self._grid_shape)
        self._nodes_flatten = grid.nodes.flatten().astype('int')
        self._nodes_at_cell = self._grid.map_node_to_cell(self._nodes_flatten).astype(int)

        # Initialize arrays
        self._zeros_at_link = self._grid.zeros(at="link")
        self._zeros_at_node = self._grid.zeros(at="node")

        # Link status
        self._inactive_links = grid.status_at_link == grid.BC_LINK_IS_INACTIVE
        self._active_links = ~self._inactive_links
        self._active_link_ids = np.arange(len(self._zeros_at_link))[self._active_links]

        # Grain size information
        self._num_grain_sizes = self._ensure_2d(grid.at_node['grains__mass']).shape[1]
        self._grain_sizes = self._ensure_2d(grid.at_node["grains_classes__size"])[grid.core_nodes[0]]

        # Physical parameters
        self._fluid_density = fluid_density
        self._sediment_density = sigma
        self._gravity = g
        self._porosity = phi
        self._submerged_specific_gravity = submerged_specific_gravity
        self._specific_gravity = sigma / fluid_density

        # Erosion and transport parameters
        self._critical_shear_stress_value = tau_crit
        self._detachment_coefficient = kr
        self._cover_depth = cover_depth_star
        self._alpha = alpha
        self._beta = beta

        # Detachment model selection
        assert detachment_model in ('shear', 'stream_power'), \
            "detachment_model must be 'shear' or 'stream_power'"
        self._detachment_model = detachment_model
        self._k_omega = k_omega
        assert 0 < q_at_node_relaxation <= 1, "q_at_node_relaxation must be in (0, 1]"
        self._q_at_node_relaxation = q_at_node_relaxation
        self._q_at_node_smoothed = None
        self._ft = ft
        assert 0 < dr_relaxation <= 1, "dr_relaxation must be in (0, 1]"
        self._dr_relaxation = dr_relaxation
        self._cv_max = cv_max
        self._cv_taper_exponent = cv_taper_exponent

        # Settling velocity parameters
        self._C1 = C1
        self._C2 = C2
        self._kinematic_viscosity = kinematic_viscosity

        # Flow geometry
        self._depression_depth = depression_depth
        self._max_deposition_slope = max_deposition_slope

        # Model options
        self._slope_field = slope
        self._update_topography = change_topo_flag

        # Bedrock grain size distribution
        self._bedrock_grain_fractions = self._ensure_2d(grid.at_node["bed_grains__proportions"])
        cumsum_fractions = np.cumsum(self._bedrock_grain_fractions[grid.core_nodes[0]])
        median_idx = int(np.argwhere(cumsum_fractions >= 0.5)[0])
        self._bedrock_median_size = self._grain_sizes[median_idx]

        if bedrock_grain_sizes is None:
            self._bedrock_grain_sizes = self._grain_sizes
        else:
            self._bedrock_grain_sizes = bedrock_grain_sizes

        # Vegetation parameters
        self._vegetation_flag = veg_flag
        self._vegetation_coefficient = vegetation_coefficient
        self._vegetation_exponent = vegetation_exponent
        self._vegetation_roughness_reference = veg_roughness_reference
        self._vegetation_cover_reference = veg_cover_reference
        self._soil_roughness = soil_roughness
        self._root_depth = root_depth
        self._omega_veg = vegetation_exponent
        self._veg_roughness_reference = veg_roughness_reference  

        # Initialize grain-size-specific arrays
        self._initialize_grain_size_arrays()

        # Calculate settling velocities
        self._calculate_settling_velocities()

        # Stability parameters
        self._min_total_load_mass = _NEGLIG
        self._max_dt = max_stable_dt
        self._stable_dt = max_stable_dt

        self._total_load_sediment_mass_at_node_per_size = self._total_load_mass_at_node
        shape_node_grains = (self._num_nodes, self._num_grain_sizes)
        self._TC = np.zeros(shape_node_grains)  # Transport capacity
        self._DR = np.zeros(shape_node_grains)  # Detachment rate
        self._c_kg = np.zeros(shape_node_grains)  # Concentration in kg

    def _initialize_grain_size_arrays(self):
        """Initialize arrays that track properties for each grain size."""
        shape_node_grains = (self._num_nodes, self._num_grain_sizes)
        shape_link_grains = (len(self._zeros_at_link), self._num_grain_sizes)

        # Critical shear stress arrays
        self._critical_shear_stress = np.full(shape_node_grains,
                                               self._critical_shear_stress_value)
        self._critical_shear_stress_initial = self._critical_shear_stress.copy()

        # Transport and flux arrays
        self._transport_capacity = np.zeros(shape_node_grains)
        self._detachment_capacity = np.zeros(shape_node_grains)
        self._net_erosion_deposition_rate = np.zeros(shape_node_grains)

        self._mass_discharge = np.zeros(shape_node_grains)

        self._net_erosion_deposition_rate_raw = np.zeros(shape_node_grains)

        self._net_erosion_deposition_rate = np.zeros(shape_node_grains)

        self._water_gradient_at_link_buffer = np.zeros_like(self._zeros_at_link)

        # Flux arrays
        self._mass_flux_at_link = np.zeros(shape_link_grains)
        self._total_load_concentration_at_link = np.zeros(shape_link_grains)
        self._total_load_mass_at_link = np.zeros(shape_link_grains)
        self._total_load_flux_at_link = np.zeros(shape_link_grains)

        # Node arrays
        self._total_load_mass_at_node = np.zeros(shape_node_grains)
        self._total_load_dzdt_at_node = np.zeros(shape_node_grains)
        self._outflux_masss_at_node = np.zeros(shape_node_grains)
        self._influx_masss_at_node = np.zeros(shape_node_grains)

        # Detachment arrays
        self._detached_soil_mass = np.zeros(shape_node_grains)
        self._detached_bedrock_mass = np.zeros(shape_node_grains)
        self._deposited_mass = np.zeros(shape_node_grains)

        # 1D node arrays
        self._detached_soil_dz = np.zeros_like(self._zeros_at_node)
        self._detached_bedrock_dz = np.zeros_like(self._zeros_at_node)
        self._deposited_dz = np.zeros_like(self._zeros_at_node)

        # Concentration arrays
        self._concentration_volume = np.ones(shape_node_grains)
        self._concentration_mass = np.ones(shape_node_grains)
        self._grain_fractions_at_node = np.zeros(shape_node_grains)
        self._total_load_fractions_at_node = np.zeros(shape_node_grains)
        self._sum_concentration_volume = np.zeros(self._num_nodes)

    def _calculate_settling_velocities(self):
        """Calculate settling velocities for each grain size class.

        The equation is from:
        Ferguson, R., & Church, M. (2004). A simple universal equation for
        grain settling velocity. Journal of Sedimentary Research, 74(6), 933-937.
        """
        self._settling_velocities = np.zeros(self._num_grain_sizes)

        for i, grain_size in enumerate(self._grain_sizes):
            numerator = (self._submerged_specific_gravity *
                        self._gravity * grain_size**2)
            denominator = (self._C1 * self._kinematic_viscosity +
                          np.sqrt(0.75 * self._C2 * self._submerged_specific_gravity *
                                 self._gravity * grain_size**3))
            self._settling_velocities[i] = numerator / denominator

    def _reset_variables(self):
        """Reset variables at the start of each timestep."""
        self._detached_soil_mass[:] = 0.0
        self._detached_bedrock_mass[:] = 0.0
        self._detached_bedrock_dz[:] = 0.0
        self._detached_soil_dz[:] = 0.0
        self._total_load_dzdt_at_node[:] = 0.0
        self._deposited_mass[:] = 0.0
        self._deposited_dz[:] = 0.0
        self._outflux_masss_at_node[:] = 0.0
        self._influx_masss_at_node[:] = 0.0
        self._total_load_concentration_at_link[:] = 0.0
        self._total_load_flux_at_link[:] = 0.0
        self._concentration_volume[:] = 1.0
        self._concentration_mass[:] = 1.0
        self._grain_fractions_at_node[:] = 0.0
        self._total_load_fractions_at_node[:] = 0.0
        self._mass_flux_at_link[:] = 0.0

        self._grid.at_node['sediment__influx'][:] = 0.0
        self._grid.at_node['sediment__outflux'][:] = 0.0

        self._transport_capacity[:] = 0.0
        self._detachment_capacity[:] = 0.0
        self._net_erosion_deposition_rate_raw[:] = 0.0

    def _map_upwind_downwind_nodes(self, elev_field='water_surface__elevation'):
        """Map upwind and downwind nodes to each link based on elev field.

        Parameters
        ----------
        elev_field : str, optional
            Name of elevation field to use for mapping. Default: 'water_surface__elevation'
        """
        elev = self._grid.at_node[elev_field]
        node_at_link_head = self._grid.node_at_link_head
        node_at_link_tail = self._grid.node_at_link_tail
        elev_head = elev[node_at_link_head]
        elev_tail = elev[node_at_link_tail]

        self._upwind_node_ids_at_link = np.where(
            elev_tail > elev_head, node_at_link_tail, node_at_link_head)
        self._downwind_node_ids_at_link = np.where(
            elev_tail < elev_head, node_at_link_tail, node_at_link_head)

    def _calculate_shear_stress(self):
        """Calculate shear stress at each node


        Returns
        -------
        shear_stress : ndarray
            Shear stress at each node [Pa]
        """
        # Get water depth and slope
        water_depth = self.grid.at_node['surface_water__depth']
        slope = self._grid.at_node[self._slope_field]

        max_downwind_gradient = self._grid.at_node['downwind__link_gradient']

        # Pre-compute constant
        rho_g = self._fluid_density * self._gravity

        # Calculate base shear stress: = ρ * g * h * S
        shear_stress = np.where(
            max_downwind_gradient > 0,
            rho_g * water_depth * slope,
            0
        )

        shear_stress = shear_stress * self._ft

        # Apply vegetation effects if enabled
        if self._vegetation_flag > 0:
            shear_stress = self._apply_vegetation_effects(shear_stress)

        return shear_stress

    def _apply_vegetation_effects(self, shear_stress):
        """Factor shear stress based on vegetation cover following the approach from:
        Istanbulluoglu, E., & Bras, R. L. (2005). Vegetation‐modulated landscape
        evolution: Effects of vegetation on landscape processes, drainage density,
        and topography. Journal of Geophysical Research, 110(F2).

        Parameters
        ----------
        shear_stress : ndarray
            Base shear stress [Pa]

        Returns
        -------
        ndarray
            Modified shear stress [Pa]
        """
        vegetation_cover = self._grid.at_node['vegetation__cover_fraction']

        # Update roughness based on vegetation cover
        vegetation_roughness = (self._vegetation_roughness_reference *
                               (vegetation_cover / self._vegetation_cover_reference) **
                               self._vegetation_exponent)

        # Partition of shear stress to soil surface
        # based on roughness partitioning approach
        fraction_to_soil = (self._soil_roughness /
                           (vegetation_roughness + self._soil_roughness)) ** 1.5

        return shear_stress * fraction_to_soil

    def _calculate_flow_width(self,
                              water_depth):
        """Calculate effective flow width in each cell.

        Flow width is approximated based on the relationship between surface
        runoff height and maximum depression capacity, following for example:
        Nunes, J. P., Vieira, G. N., Seixas, J., Goncalves, P., & Carvalhais, N.
        (2005). Evaluating the MEFIDIS model for runoff and soil erosion
        prediction during rainfall events. Catena, 61(2-3), 210-228.

        Parameters
        ----------
        water_depth : ndarray
            Water depth at each node [m]

        Returns
        -------
        ndarray
            Effective flow width at each node [m]
        """
        # Flow width proportional to water depth relative to depression depth
        flow_width = water_depth / self._depression_depth

        # Cap flow width at grid cell width
        flow_width = np.minimum(flow_width, self._dx)

        return flow_width

    def _calculate_transport_capacity(self,
                                      shear_stress
                                      ):
        """Calculate sediment transport capacity
        Parameters
        ----------
        shear_stress : ndarray
            Shear stress at each node [Pa]
        Returns
        -------
        ndarray
            Transport capacity for each grain size at each node [kg/(m²·s)]
        """
        # Get median grain size at each node
        median_sizes = self._grid.at_node['median_size__mass']

        # Use where to handle zeros without modifying original array
        median_sizes_safe = np.where(
            median_sizes == 0,
            self._bedrock_median_size,
            median_sizes
        )

        shape = (len(self._grid.core_nodes), self._num_grain_sizes)

        # Pre-compute constant once (cache if called multiple times)
        if not hasattr(self, '_const_sg_g_rho'):
            self._const_sg_g_rho = (self._fluid_density *
                                    (self._specific_gravity - 1) *
                                    self._gravity)

        self._transport_capacity = cfuncs_ErosionDeposition.calc_TC(
            self._alpha,
            self._beta,
            median_sizes_safe,
            self._ensure_2d(self._grid.at_node['grains_classes__size']),
            shear_stress,
            self._transport_capacity,
            self._specific_gravity,
            self._fluid_density,
            self._grid.core_nodes,
            self._const_sg_g_rho,
            shape
        )

        # Smoothly choke off transport capacity if approaches the physical packing limit
        # cv_max.
        cv_taper = 1.0 - (self._sum_concentration_volume / self._cv_max) ** self._cv_taper_exponent
        np.clip(cv_taper, 0.0, 1.0, out=cv_taper)
        self._transport_capacity *= cv_taper[:, np.newaxis]

        return self._transport_capacity

    def _calculate_detachment_capacity(self, 
                                       shear_stress,
                                       q_at_node):
        """Calculate detachment capacity.

        Two models are supported, selected via `self._detachment_model`:

        - 'shear' (default): excess shear stress approach, Dc = kr * (tau_s - tau_crit)
        - 'stream_power': RHEM V2.3 (Al-Hamdan et al., 2012b) concentrated-flow
          detachment, Dc = k_omega * omega, omega = rho * g * S * q (stream
          power, kg/s^3).

        Parameters
        ----------
        shear_stress : ndarray
            Shear stress at each node [Pa]
        q_at_node : ndarray, optional
            Water discharge at each node [m^3/s].

        Returns
        -------
        ndarray
            Detachment capacity for each grain size [kg/(m²·s)]
        """
        shape = (len(self._grid.core_nodes), self._num_grain_sizes)

        if self._detachment_model == 'shear':
            detachment_capacity = cfuncs_ErosionDeposition.calc_Dc(
                shear_stress,
                self._critical_shear_stress,
                self._grid.core_nodes,
                self._detachment_capacity,  # Reuse preallocated array
                self._detachment_coefficient,
                shape
            )

        elif self._detachment_model == 'stream_power':

            slope = self._grid.at_node[self._slope_field]
            max_downwind_gradient = self._grid.at_node['downwind__link_gradient']
            slope_masked = np.where(max_downwind_gradient > 0, slope, 0)
            stream_power = self._fluid_density * self._gravity * slope_masked * q_at_node * self._ft
            detachment_capacity = cfuncs_ErosionDeposition.calc_Dc_stream_power(
                stream_power,
                self._grid.core_nodes,
                self._detachment_capacity,  # Reuse preallocated array
                self._k_omega,
                shape
            )

        else:
            raise ValueError(
                "detachment_model must be 'shear' or 'stream_power', got %r" % self._detachment_model)

        self._detachment_capacity = detachment_capacity
        return detachment_capacity

    def _calculate_net_erosion_deposition_rate(self):
        """Calculate net erosion or deposition rate at each node.

        This calculation follows Foster 1982 and determines whether each node
        experiences net erosion or deposition based on transport capacity,
        detachment capacity, flow discharge, and sediment concentration.


        Foster, G. R. (1982). Modeling the erosion process. In C. T. Haan, H. P.
        Johnson, & D. L. Brakensiek (Eds.), Hydrologic Modeling of Small
        Watersheds (pp. 297-380). ASAE Monograph No. 5, American Society of
        Agricultural Engineers.

        Returns
        -------
        ndarray
            Net erosion or deposition rate for each grain size [kg/(m²·s)]
        """
        water_depth = self.grid.at_node['surface_water__depth']

        # Calculate shear stress
        shear_stress = self._calculate_shear_stress()
        self._shear_stress = shear_stress

        # Calculate outgoing discharge at each node
        shape = len(self._active_link_ids)
        self._zeros_at_node[:] = 0.0
        discharge_out = cfuncs_ErosionDeposition.sum_out_discharge(
            self._upwind_node_ids_at_link,
            np.abs(self._grid.at_link['surface_water__discharge']),
            self._active_link_ids,
            self._zeros_at_node,  
            shape
        )

        # Exponential smoothing on discharge
        if self._q_at_node_smoothed is None:
            self._q_at_node_smoothed = discharge_out.copy()
        else:
            self._q_at_node_smoothed = (self._q_at_node_relaxation * discharge_out
                                         + (1.0 - self._q_at_node_relaxation) * self._q_at_node_smoothed)
        discharge_out = self._q_at_node_smoothed

        # Calculate transport capacity
        self._calculate_transport_capacity(shear_stress)

        # Calculate detachment capacity
        self._calculate_detachment_capacity(shear_stress, discharge_out)

        # Calculate total sediment flux
        shape = (len(self._grid.core_nodes), self._num_grain_sizes)
        mass_discharge = cfuncs_ErosionDeposition.calc_CQ(
            self._concentration_mass,
            self._mass_discharge,
            discharge_out,
            self._grid.core_nodes,
            shape,
            int(self._dx)
        )


        # Calculate effective flow width
        flow_width = self._calculate_flow_width(water_depth)

        # Calculate net erosion/deposition rate
        self._net_erosion_deposition_rate_raw = cfuncs_ErosionDeposition.calc_DR(
            flow_width,
            mass_discharge,
            self._transport_capacity,
            self._detachment_capacity,
            self._settling_velocities,
            self._grid.core_nodes,
            discharge_out,
            self._net_erosion_deposition_rate_raw,  # Reuse preallocated array
            self._dx,
            shape
        )

        self._net_erosion_deposition_rate *= (1.0 - self._dr_relaxation)
        self._net_erosion_deposition_rate += self._dr_relaxation * self._net_erosion_deposition_rate_raw

        water_depth_safe = np.maximum(water_depth, _NEGLIG)
        water_depth_expanded = np.expand_dims(water_depth_safe, -1)

        np.divide(
            self._total_load_mass_at_node,
            self._sediment_density * water_depth_expanded * self._dx_squared,
            out=self._concentration_volume  # Reuse existing array
        )
        sum_concentration = np.sum(self._concentration_volume, axis=1)
        oversaturated = sum_concentration >= 1
        self._net_erosion_deposition_rate[:] = self._net_erosion_deposition_rate
        self._net_erosion_deposition_rate[oversaturated, :] = -np.inf

    def _calculate_sediment_flux_at_links(self):
        """Calculate sediment flux through links and resulting changes at nodes."""
        # Get water discharge and gradients
        water_discharge = self.grid.at_link['surface_water__discharge']

        water_gradient = self._grid.calc_grad_at_link(
            'water_surface__elevation', out=self._water_gradient_at_link_buffer)

        # Calculate water depth at nodes (avoid division by zero)
        water_depth = self.grid.at_node['surface_water__depth']
        water_depth_safe = np.maximum(water_depth, _NEGLIG)
        water_depth_expanded = np.expand_dims(water_depth_safe, -1)

        # Calculate sediment concentration at links (from upwind nodes)
        self._total_load_concentration_at_link[:] = np.divide(
            self._total_load_mass_at_node[self._upwind_node_ids_at_link, :],
            water_depth_expanded[self._upwind_node_ids_at_link, :] * self._dx_squared
        )

        # Determine flow direction at nodes
        outlinks_at_node = self.grid.link_at_node_is_downwind(water_gradient)

        # Calculate mass flux at each link
        shape = [len(self._active_link_ids), self._num_grain_sizes]
        self._mass_flux_at_link[:] = cfuncs_ErosionDeposition.calc_flux_at_link_per_size(
            water_discharge,
            self._total_load_concentration_at_link,
            self._grid.active_links,
            self._mass_flux_at_link,  # Reuse instead of creating zeros
            shape
        )

        np.abs(self._mass_flux_at_link, out=self._mass_flux_at_link)

        # Calculate in/out fluxes at nodes
        self._calculate_influx_outflux_at_nodes(outlinks_at_node)

        # Convert mass fluxes to volume fluxes
        shape = self._total_load_flux_at_link.shape
        self._total_load_flux_at_link[:] = cfuncs_ErosionDeposition.calc_flux_at_link(
            self._dx,
            self._sediment_density,
            self._porosity,
            np.abs(self._mass_flux_at_link),
            -np.sign(water_gradient),
            self._total_load_flux_at_link,
            shape
        )

        # Calculate divergence of sediment flux
        self._calculate_flux_divergence()

    def _calculate_influx_outflux_at_nodes(self, outlinks_at_node):
        """Calculate sediment influx and outflux at each node.

        Parameters
        ----------
        outlinks_at_node : ndarray
            Boolean array indicating which links are outgoing from each node
        """
        shape = [len(self._active_link_ids), self._num_grain_sizes]
        total_outflux = np.zeros_like(self._zeros_at_node)
        total_influx = np.zeros_like(self._zeros_at_node)

        # Calculate fluxes
        (self._outflux_masss_at_node[:],
         self._influx_masss_at_node[:],
         total_outflux,
         total_influx) = cfuncs_ErosionDeposition.get_outin_fluxes(
            self._upwind_node_ids_at_link,
            self._downwind_node_ids_at_link,
            self._mass_flux_at_link,
            self._active_link_ids,
            self._outflux_masss_at_node,
            self._influx_masss_at_node,
            total_outflux,
            total_influx,
            shape
        )

        # Check for mass conservation violations (outflux > available mass)
        indices_to_correct = np.where(
            self._outflux_masss_at_node > self._total_load_mass_at_node
        )

        if np.any(indices_to_correct[0]):
            # Calculate correction ratios
            ratios = np.divide(
                self._total_load_mass_at_node[indices_to_correct],
                self._outflux_masss_at_node[indices_to_correct]
            )

            # Apply corrections to mass flux at links
            for i, (node_id, grain_size_id) in enumerate(
                zip(indices_to_correct[0], indices_to_correct[1])
            ):
                outgoing_links = self._grid.links_at_node[node_id, :][
                    outlinks_at_node[node_id, :]
                ]
                self._mass_flux_at_link[outgoing_links, grain_size_id] *= ratios[i]

            # Recalculate fluxes with corrected values
            self._outflux_masss_at_node.fill(0.0)
            self._influx_masss_at_node.fill(0.0)
            total_outflux.fill(0.0)
            total_influx.fill(0.0)

            (self._outflux_masss_at_node[:],
             self._influx_masss_at_node[:],
             total_outflux,
             total_influx) = cfuncs_ErosionDeposition.get_outin_fluxes(
                self._upwind_node_ids_at_link,
                self._downwind_node_ids_at_link,
                self._mass_flux_at_link,
                self._active_link_ids,
                self._outflux_masss_at_node,
                self._influx_masss_at_node,
                total_outflux,
                total_influx,
                shape
            )

        # Store total fluxes
        self._grid.at_node['sediment__influx'][:] = total_influx
        self._grid.at_node['sediment__outflux'][:] = total_outflux

    def _calculate_flux_divergence(self):
        """Calculate flux divergence at each node for each grain size."""
        # Find grain sizes with active transport
        active_grain_sizes = np.where(np.any(self._total_load_flux_at_link, axis=0))[0]

        if len(active_grain_sizes) == 0:
            return

        # Calculate divergence for each active grain size
        for grain_size_id in active_grain_sizes:
            dzdt = self._calculate_dzdt_for_grain_size(grain_size_id)
            self._total_load_dzdt_at_node[:, grain_size_id] = dzdt

    def _calculate_dzdt_for_grain_size(self, grain_size_id):
        """Calculate rate of elevation change according to change in elev of each grain size class.

        Parameters
        ----------
        grain_size_id : int
            Index of the grain size class

        Returns
        -------
        ndarray
            Rate of elevation change at each node [m/s]
        """
        flux = self._total_load_flux_at_link[:, grain_size_id]
        dzdt = np.zeros_like(self._zeros_at_node)

        dzdt = cfuncs_ErosionDeposition.calc_flux_div_at_node(
            self._grid_shape,
            self._xy_spacing,
            flux,
            dzdt
        )

        return -dzdt

    def _partition_erosion_deposition(self):
        """Partition net erosion/deposition between soil, bedrock, and load."""
        # Get soil and grain mass information
        soil_depth = self._grid.at_node['soil__depth']
        grain_masss = self._ensure_2d(self.grid.at_node['grains__mass'])
        grain_masss = np.maximum(grain_masss, _NEGLIG)

        # Calculate soil cover exponential factor
        soil_cover_factor = 1 - np.exp(-soil_depth / self._cover_depth)

        # Calculate grain fractions
        total_grain_mass = np.sum(grain_masss, axis=1)
        shape = grain_masss.shape
        self._grain_fractions_at_node[:] = cfuncs_ErosionDeposition.calc_concentration(
            np.ones_like(grain_masss),
            grain_masss,
            total_grain_mass,
            shape
        )

        # Calculate total_load sediment fractions
        total_load_mass_positive = self._total_load_mass_at_node.copy()
        total_load_mass_positive = np.maximum(total_load_mass_positive, 1e-10)
        total_total_load_mass = np.sum(total_load_mass_positive, axis=1)

        self._total_load_fractions_at_node[:] = cfuncs_ErosionDeposition.calc_concentration(
            np.ones_like(total_load_mass_positive),
            total_load_mass_positive,
            total_total_load_mass,
            shape
        )

        # Partition erosion/deposition
        core_nodes = self._grid.core_nodes
        shape_core = (len(core_nodes), self._num_grain_sizes)
        mass_to_dz_factor = (self._sediment_density *
                              (1 - self._porosity) *
                              self._dx_squared)

        (self._detached_soil_mass[:],
         self._detached_bedrock_mass[:],
         self._detached_soil_dz[:],
         self._detached_bedrock_dz[:],
         self._deposited_mass[:],
         self._deposited_dz[:]) = cfuncs_ErosionDeposition.calc_detached_deposited(
            self._net_erosion_deposition_rate,
            np.abs(self._net_erosion_deposition_rate),
            grain_masss,
            self._grain_fractions_at_node,
            self._detached_soil_mass,
            self._detached_bedrock_mass,
            self._total_load_fractions_at_node,
            self._bedrock_grain_fractions,
            total_load_mass_positive,
            self._deposited_mass,
            self._deposited_dz,
            self._detached_soil_dz,
            self._detached_bedrock_dz,
            soil_cover_factor,
            core_nodes,
            mass_to_dz_factor,
            mass_to_dz_factor,
            shape_core,
            self._dx
        )

    def _calculate_stable_timestep(self):
        """Calculate stable timestep based on CFL-like conditions.

        Considers five stability conditions:
        1. Erosion should not exceed half the downwind gradient
        2. Deposition flux should not exceed available total_load sediment
        3. Deposition should not create excessive topographic inversions
        4. Outgoing flux should not exceed available sediment mass
        5. Soil detachment should not exceed available soil mass per grain
           size (grains__mass)

        Returns
        -------
        float
            Maximum stable timestep [s]
        """
        max_downwind_gradient = self._grid.at_node['downwind__link_gradient']
        max_upwind_gradient = self._grid.at_node['upwind__link_gradient']
        grain_masss = self._ensure_2d(self.grid.at_node['grains__mass'])

        (dt_erosion, dt_deposition_mass, dt_deposition_topo, dt_mass,
         dt_erosion_mass) = cfuncs_ErosionDeposition.calc_stable_dt(
            max_downwind_gradient,
            max_upwind_gradient,
            self._detached_bedrock_dz,
            self._detached_soil_dz,
            self._deposited_dz,
            self._total_load_mass_at_node,
            self._deposited_mass,
            self._total_load_dzdt_at_node,
            self._outflux_masss_at_node,
            self._dx,
            self._dx_squared,
            self._sediment_density,
            self._porosity,
            self._max_deposition_slope,
            self._min_total_load_mass,
            self._total_load_mass_at_node.shape,
            grain_masss,
            self._detached_soil_mass,
        )

        # Take minimum of all stability conditions
        self._stable_dt = min(dt_erosion, dt_deposition_mass,
                             dt_deposition_topo, dt_mass, dt_erosion_mass,
                             self._max_dt)

        # Store individual components for diagnostics
        self._dt_erosion = dt_erosion
        self._dt_deposition_mass = dt_deposition_mass
        self._dt_deposition_topo = dt_deposition_topo
        self._dt_mass = dt_mass
        self._dt_erosion_mass = dt_erosion_mass

    def calc_rates(self):
        """Calculate erosion and deposition rates."""
        # Reset variables
        self._reset_variables()

        # Early exit if no water depth (no flow)
        max_water_depth = np.max(self.grid.at_node['surface_water__depth'])
        if max_water_depth < _NEGLIG:
            # No flow. Bound the stable timestep by max_stable_dt
            self._stable_dt = self._max_dt
            return

        # Map flow directions
        self._map_upwind_downwind_nodes()

        # Calculate concentrations
        self._update_sediment_concentrations()

        # Calculate erosion/deposition rates
        self._calculate_net_erosion_deposition_rate()

        # Partition into soil/bedrock/total_load
        self._partition_erosion_deposition()

        # Calculate sediment fluxes
        self._calculate_sediment_flux_at_links()

        # Calculate stable timestep
        self._calculate_stable_timestep()

        # Update legacy diagnostic arrays for backward compatibility
        self._TC[:] = self._transport_capacity
        self._DR[:] = self._net_erosion_deposition_rate
        self._c_kg[:] = self._concentration_mass

    def _update_sediment_concentrations(self):
        """Update volumetric and mass concentrations of total_load sediment."""

        water_depth = self.grid.at_node['surface_water__depth']

        water_depth_safe = np.maximum(water_depth, _NEGLIG)
        water_depth_expanded = np.expand_dims(water_depth_safe, -1)

        # Pre-compute denominator once using cached dx_squared
        water_volume = water_depth_expanded * self._dx_squared

        # Volumetric and mass concentration.
        total_load_volume = np.maximum(self._total_load_mass_at_node / self._sediment_density, 0)
        np.divide(total_load_volume, water_volume, out=self._concentration_volume)

        total_load_mass_positive = np.maximum(self._total_load_mass_at_node, 0)
        np.divide(total_load_mass_positive, water_volume, out=self._concentration_mass)

        # Total (summed across grain sizes) volumetric concentration per node
        np.sum(self._concentration_volume, axis=1, out=self._sum_concentration_volume)

    def run_one_step(self, dt=None):
        """Advance topography and sediment distribution by a total time dt.

        Parameters
        ----------
        dt : float, optional
            Total timestep duration to advance by [s]. Default: 1.0

        """
        elapsed = 0.0
        n_substeps = 0

        while elapsed < dt:
            if dt is None or elapsed>0 :
                # Rates (and stable_dt) depend on the current state:
                self.calc_rates()

            remaining = dt - elapsed
            sub_dt = self._stable_dt
            if not np.isfinite(sub_dt) or sub_dt <= 0:
                # Shouldn't normally happen
                sub_dt = remaining
            sub_dt = min(sub_dt, remaining)

            self._apply_step(sub_dt)

            elapsed += sub_dt
            n_substeps += 1

    def _apply_step(self, dt):
        """Apply one stable sub-step using the rates from calc_rates().

        Parameters
        ----------
        dt : float
            Sub-step duration [s].
        """
        # Get references to grid fields
        grain_masss = self._ensure_2d(self.grid.at_node['grains__mass'])
        soil_depth = self._grid.at_node['soil__depth']
        bedrock_elevation = self._grid.at_node['bedrock__elevation']
        topographic_elevation = self._grid.at_node['topographic__elevation']
        total_load_mass_total = self.grid.at_node['total_load__sediments_mass']

        # Calculate net flux of total_load sediment from adjacent nodes
        net_total_load_flux = (self._total_load_dzdt_at_node *
                             self._dx_squared *
                             self._sediment_density *
                             (1 - self._porosity) *
                             dt)

        # Calculate deposition
        deposition_mass = self._deposited_mass * dt
        deposition_mass = np.minimum(
            deposition_mass,
            self._total_load_mass_at_node
        )

        # Calculate detachment
        detachment_mass = (self._detached_soil_mass +
                            self._detached_bedrock_mass) * dt

        # Update total_load sediment
        self._total_load_mass_at_node[:] += detachment_mass + net_total_load_flux
        self._total_load_mass_at_node[:] -= deposition_mass

        np.maximum(self._total_load_mass_at_node, 0, out=self._total_load_mass_at_node)

        total_load_mass_total[:] = np.sum(self._total_load_mass_at_node, axis=1)

        # Update topography if enabled
        if self._update_topography:
            # Net change in soil layer mass
            net_mass_change = deposition_mass - self._detached_soil_mass * dt

            # Update grain masss (per unit area)
            grain_masss[:] += net_mass_change / self._dx_squared

            np.maximum(grain_masss, 0, out=grain_masss)

            # Update soil depth
            total_grain_mass = np.sum(grain_masss, axis=1)
            soil_depth[:] = (total_grain_mass / self._sediment_density /
                           (1 - self._porosity))

            # Update bedrock elevation (lowered by detachment)
            bedrock_elevation[:] -= self._detached_bedrock_dz * dt

            # Update topographic elevation
            topographic_elevation[:] = soil_depth + bedrock_elevation

        # Update vegetation if enabled
        if self._vegetation_flag > 0:
            self._update_vegetation(dt, deposition_mass, detachment_mass)

    def _update_vegetation(self,
                           dt,
                           deposition_mass,
                           detachment_mass):
        """Update vegetation based on erosion/deposition.

        Parameters
        ----------
        dt : float
            Timestep [s]
        deposition_mass : ndarray
            Deposited sediment mass [kg]
        detachment_mass : ndarray
            Detached sediment mass [kg]
        """
        vegetation_cover = self._grid.at_cell['vegetation__cover_fraction']

        # Erosion-depth-based vegetation removal
        self._update_vegetation_by_erosion(
            dt, vegetation_cover, deposition_mass, detachment_mass
        )

        # Update roughness based on new vegetation cover
        self._update_roughness_from_vegetation()

    def _update_vegetation_by_erosion(self, dt, vegetation_cover,
                                     deposition_mass, detachment_mass):
        """Update vegetation based on net erosion depth.

        Parameters
        ----------
        dt : float
            Timestep [s]
        vegetation_cover : ndarray
            Vegetation cover fraction at cells [-]
        deposition_mass : ndarray
            Deposited mass [kg]
        detachment_mass : ndarray
            Detached mass [kg]
        """
        # Calculate net mass change
        net_mass_change = (deposition_mass -
                            (self._detached_soil_mass * dt +
                             self._detached_bedrock_mass * dt))

        total_mass_change = np.sum(net_mass_change, axis=1)

        # Convert to depth change
        depth_change = np.abs(total_mass_change) / (
            self._sediment_density * self._dx_squared * (1 - self._porosity)
        )

        # Map to cells
        depth_change_at_cell = self._grid.map_node_to_cell(depth_change)

        # Calculate vegetation removal based on root depth
        root_depth = self._root_depth  # [m]
        vegetation_removal = ((depth_change_at_cell / root_depth) *
                             vegetation_cover)
        vegetation_removal = np.minimum(
            vegetation_removal,
            vegetation_cover - _TEN_MINUS_THREE
        )

        # Calculate removal fraction
        removal_fraction = np.divide(
            vegetation_removal,
            vegetation_cover,
            where=vegetation_cover > _TEN_MINUS_THREE,
            out=np.zeros_like(vegetation_removal)
        )

        retention_fraction = 1 - removal_fraction

        # Apply reduction
        self._apply_vegetation_reduction(retention_fraction)

    def _apply_vegetation_reduction(self, retention_fraction):
        """Apply vegetation reduction due to erosion

        Parameters
        ----------
        retention_fraction : ndarray
            Fraction of vegetation remaining at each cell [-]
        """
        if np.any(retention_fraction):
            self._grid.at_cell['vegetation__live_biomass'][:] *= retention_fraction
            self._grid.at_cell['vegetation__dead_biomass'][:] *= retention_fraction
            self._grid.at_cell['vegetation__live_leaf_area_index'][:] *= retention_fraction
            self._grid.at_cell['vegetation__dead_leaf_area_index'][:] *= retention_fraction

            vegetation_cover = self._grid.at_cell['vegetation__cover_fraction']
            vegetation_cover[:] *= retention_fraction
            vegetation_cover[:] = np.maximum(vegetation_cover, 1e-8)

        # Map back to nodes
        vegetation_cover_at_node = self._grid.at_node['vegetation__cover_fraction']
        vegetation_cover_at_node[self._nodes_at_cell] = (
            self._grid.at_cell['vegetation__cover_fraction'][:]
        )

    def _update_roughness_from_vegetation(self):
        """Update Manning's n based on vegetation cover."""
        vegetation_cover = self._grid.at_node['vegetation__cover_fraction']

        #calculate vegetation roughness
        vegetation_roughness = (
            self._vegetation_roughness_reference *
            (vegetation_cover / self._vegetation_cover_reference) **
            self._vegetation_exponent
        )

        # Combine soil and vegetation roughness
        bare_soil_fraction = 1 - vegetation_cover
        self._grid.at_node['mannings_n'][:] = (
            bare_soil_fraction * self._soil_roughness +
            vegetation_cover * vegetation_roughness
        )

        # Map mannigns to links
        self._grid.at_link['mannings_n'][:] = (
            self._grid.at_node['mannings_n'][self._upwind_node_ids_at_link]
        )

    def update_vegetated_roughness(self):
        """Updating Manning's n and roughness based on vegetation cover.
        """

        if 'vegetation__cover_fraction' in self._grid.at_node:
            vegetation_cover = self._grid.at_node['vegetation__cover_fraction']
        elif 'vegetation_cover__fraction' in self._grid.at_node:
            vegetation_cover = self._grid.at_node['vegetation_cover__fraction']
        else:
            # No vegetation, use soil roughness only
            self._grid.at_link['mannings_n'][:] = self._soil_roughness
            self._ft = 1.0
            return

        if hasattr(self, '_omega_veg'):
            omega_veg = self._omega_veg
        else:
            omega_veg = self._vegetation_exponent

        # Legacy calculation using reference cover
        if hasattr(self, '_veg_roughness_reference'):
            veg_roughness_ref = self._veg_roughness_reference
        else:
            veg_roughness_ref = self._vegetation_roughness_reference

        # Calculate roughness
        vegetation_roughness = veg_roughness_ref * vegetation_cover ** omega_veg

        # Combine soil and vegetation roughness
        bare_soil_fraction = 1 - vegetation_cover
        self._grid.at_node['mannings_n'][:] = (
            bare_soil_fraction * self._soil_roughness +
            vegetation_cover * vegetation_roughness
        )

        # Map to links (use upwind node value)
        self._grid.at_link['mannings_n'][:] = (
            self._grid.at_node['mannings_n'][self._upwind_node_ids_at_link]
        )

        self._ft = (self._soil_roughness / (vegetation_roughness + self._soil_roughness)) ** 1.5

    def _get_median_grain_indices(self):
        """Get indices of median grain size at each node.

        Returns
        -------
        ndarray
            Index of median grain size at each node
        """
        grain_sizes_at_nodes = self._ensure_2d(self._grid.at_node['grains_classes__size'])[
            self._grid.core_nodes, :
        ]
        median_sizes_at_nodes = self._grid.at_node['median_size__mass'][
            self._grid.core_nodes, np.newaxis
        ]

        matches = grain_sizes_at_nodes == median_sizes_at_nodes
        median_indices = np.where(matches)[1]

        return median_indices

    @property
    def stable_dt(self):
        """Return the maximum stable timestep [s]."""
        return self._stable_dt

    @property
    def total_load_mass_at_node(self):
        """Return total_load sediment mass at node (for each grain size [kg/m2])."""
        return self._total_load_mass_at_node
