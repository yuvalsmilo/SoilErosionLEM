# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 10:35:37 2022

@author: yuvalshm
"""
import matplotlib.pyplot as plt
import numpy as np
import copy
from landlab import RasterModelGrid
from landlab.grid.nodestatus import NodeStatus
import numpy as np
from landlab import Component
import time


class SlabFailures(Component):
    _name = "SlabFailures"
    _unit_agnostic = True

    _info = {
        "topographic__elevation": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Land surface topographic elevation",
        },
        "topographic__steepest_slope": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "The steepest *downhill* slope",
        },
        "flow__receiver_node": {
            "dtype": int,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "Node array of receivers (node that receives flow from current node)",
        },
        "flow__upstream_node_order": {
            "dtype": int,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "Node array containing downstream-to-upstream ordered list of node IDs",
        },
        # Note that this field has to be provided in addition to the \
        # flow__receiver_node and will be used to route sediments over the hillslope
        "hill_flow__receiver_node": {
            "dtype": int,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "Node array of receivers (node that receives flow from current node)",
        },
        # Note that this field has to be provided in addition to the \
        # flow__receiver_proportions and will be used to route sediments
        # over the hillslope
        "hill_flow__receiver_proportions": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "Node array of proportion of flow sent to each receiver.",
        },
        "hill_topographic__steepest_slope": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "The steepest *downhill* slope",
        },
        "soil__depth": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Soil depth at node",
        },
        "landslide_sediment_point_source": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Source of sediment",
        },
        'landslide__deposition': {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Landslide deposition at node",
        },
        "landslide_soil_sediment_point_source": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Source of sediment",
        },
        "landslide_bedrock_sediment_point_source": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Source of sediment",
        },
        'landslide__deposition_soil': {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Landslide deposition at node",
        },
        'landslide__deposition_bedrock': {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Landslide deposition at node",
        },
    }

    def __init__(self,
                 grid,
                 angle_int_frict=1.0,
                 threshold_slope=None,
                 cohesion_eff=1e4,
                 rho_r=2700,
                 grav=9.81,
                 fraction_fines_LS=0,
                 phi=0,
                 seed=2021,
                 landslides_on_boundary_nodes=True,
                 critical_sliding_nodes=None,
                 min_deposition_slope=0,
                 max_dist = 10,
                 failure_angle = 0.3,
                 bedrock_sediment_grainsizes = None,
                 ):
        super(SlabFailures, self).__init__(grid)

        topo = self.grid.at_node["topographic__elevation"]
        soil = self.grid.at_node["soil__depth"]

        if "bedrock__elevation" not in grid.at_node:
            grid.add_field("bedrock__elevation", topo - soil, at="node", dtype=float)

        # Check consistency of bedrock, soil and topographic elevation fields
        if not np.allclose(
                grid.at_node["bedrock__elevation"][self.grid.core_nodes] + grid.at_node["soil__depth"][self.grid.core_nodes],
                grid.at_node["topographic__elevation"][self.grid.core_nodes],
        ):
            raise RuntimeError(
                "The sum of bedrock elevation and topographic elevation should be equal"
            )

        self.initialize_output_fields()

        # Store grid and parameters
        self._angle_int_frict = angle_int_frict
        if threshold_slope is None:
            self._threshold_slope = angle_int_frict
        else:
            self._threshold_slope = threshold_slope
        self._cohesion_eff = cohesion_eff
        self._rho_r = rho_r
        self._grav = grav
        self._phi = phi
        self._max_distance_of_slide_from_crit_node = max_dist
        self._landslides_on_boundary_nodes = landslides_on_boundary_nodes
        self._critical_sliding_nodes = critical_sliding_nodes
        self._min_deposition_slope = min_deposition_slope
        self._zeros_at_node = self._grid.zeros(at="node")
        self._zeros_at_link = self._grid.zeros(at="link")
        self._zeros_at_link_for_fractions = np.zeros(
            (np.shape(self._zeros_at_link)[0], np.shape(self._grid.at_node['grains__weight'])[1]))
        self._sum_dzdt = self._zeros_at_node.copy()
        self._failure_angle = failure_angle

        self._landslides_size = []
        self._landslides_volume = []
        self._landslides_volume_sed = []
        self._landslides_volume_bed = []

        if bedrock_sediment_grainsizes ==None:
            self._bedrock_sediment_grainsizes = self._grid.at_node["bed_grains__proportions"][self._grid.core_nodes[0]]
        # Check input values
        if phi >= 1.0 or phi < 0.0:
            raise ValueError(f"Porosity must be between 0 and 1 ({phi})")

        if fraction_fines_LS > 1.0 or fraction_fines_LS < 0.0:
            raise ValueError(
                f"Fraction of fines must be between 0 and 1 ({fraction_fines_LS})"
            )

        # Set seed
        if seed is not None:
            np.random.seed(seed)

    def failures_erosion(self):
        """

        """
        # Pointers
        topo = self.grid.at_node["topographic__elevation"]
        bed = self.grid.at_node["bedrock__elevation"]
        steepest_slope = self.grid.at_node["topographic__steepest_slope"]
        soil_d = self.grid.at_node["soil__depth"]
        grain_fractions = self.grid.at_node['grains__weight']
        before_topo_copy = np.copy(topo)
        before_soil_copy = np.copy(soil_d)

        landslide_sed_in_bedrock = self._grid.at_node["landslide_bedrock_sediment_point_source"]
        landslide_sed_in_bedrock.fill(0.)
        landslide_sed_in_soil = self._grid.at_node["landslide_soil_sediment_point_source"]
        landslide_sed_in_soil.fill(0)
        soil_removal = self._grid.at_node["landslide__deposition_soil"]
        soil_removal.fill(0.)


        # Reset data structures to store properties of simulated landslides.
        self._landslides_size = []
        self._landslides_volume = []
        self._landslides_volume_sed = []
        self._landslides_volume_bed = []


        if self._critical_sliding_nodes is None:
            
            # Calculate gradients
            height_cell = topo - topo[self.grid.at_node["flow__receiver_node"]]

            angle_int_frict_radians = np.arctan(self._angle_int_frict)
            height_critical = np.divide(
                (4 * self._cohesion_eff / (self._grav * self._rho_r))
                * (np.sin(np.arctan(steepest_slope)) * np.cos(angle_int_frict_radians)),
                1 - np.cos(np.arctan(steepest_slope) - angle_int_frict_radians),
                where=(1 - np.cos(np.arctan(steepest_slope) - angle_int_frict_radians))
                      > 0,
                out=np.zeros_like(steepest_slope),
            )

            above_critical_height = np.divide(
                height_cell,
                height_critical,
                where=height_critical > 0,
                out=np.zeros_like(height_critical),
            )
            above_critical_height[np.arctan(steepest_slope) <= angle_int_frict_radians] = 0
            critical_landslide_nodes = np.unique(
                 self.grid.at_node["flow__receiver_node"][np.where(above_critical_height > 1)])


            # Remove boundary nodes
            if not self._landslides_on_boundary_nodes:
                critical_landslide_nodes = critical_landslide_nodes[
                    ~self.grid.node_is_boundary(critical_landslide_nodes)
                ]
        else:
            critical_landslide_nodes = np.array(self._critical_sliding_nodes)

        while critical_landslide_nodes.size > 0:
            
            crit_node = critical_landslide_nodes[0]  # start at first critical node
            crit_node_el = topo[crit_node]


            # get 8 neighbors and only keep those to active nodes which are upstream
            neighbors = np.concatenate(
                (
                    self.grid.active_adjacent_nodes_at_node[crit_node],
                    self.grid.diagonal_adjacent_nodes_at_node[crit_node],
                )
            )

            neighbors = neighbors[neighbors != -1]
            neighbors_up = neighbors[topo[neighbors] > crit_node_el]

            x_crit_node = self.grid.node_x[crit_node]
            y_crit_node = self.grid.node_y[crit_node]

            dist_to_initial_node = np.sqrt(
                np.add(
                    np.square(x_crit_node - self.grid.node_x[neighbors_up]),
                    np.square(y_crit_node - self.grid.node_y[neighbors_up]),
                )
            )
            slope_neighbors_to_crit_node = (
                                                   topo[neighbors_up] - crit_node_el
                                           ) / dist_to_initial_node

            neighbors_up = neighbors_up[
                (slope_neighbors_to_crit_node > self._angle_int_frict) & (dist_to_initial_node < self._max_distance_of_slide_from_crit_node)]

            slope_neighbors_to_crit_node = slope_neighbors_to_crit_node[
                (slope_neighbors_to_crit_node > self._angle_int_frict) & (dist_to_initial_node<self._max_distance_of_slide_from_crit_node)
                ]

            if slope_neighbors_to_crit_node.size > 0:
                slope_slide = max(slope_neighbors_to_crit_node)
                store_volume_bed = 0.0
                store_volume_sed = 0.0
                upstream_count = 0
                upstream_neighbors = neighbors_up
                if not self._landslides_on_boundary_nodes:
                    upstream_neighbors = upstream_neighbors[
                        ~self.grid.node_is_boundary(upstream_neighbors)
                    ]
                
                nb_landslide_cells = 0
                # If landslides become unrealistically big, exit algorithm
                while upstream_neighbors.size > 0 and (
                        upstream_count <= self._max_pixelsize_landslide
                        and nb_landslide_cells < 1e5
                ):
                    distance_to_crit_node = np.sqrt(
                        np.add(
                            np.square(
                                x_crit_node - self.grid.node_x[upstream_neighbors[0]]
                            ),
                            np.square(
                                y_crit_node - self.grid.node_y[upstream_neighbors[0]]
                            ),
                        )
                    )
                    new_el = crit_node_el + distance_to_crit_node * self._failure_angle 
                    nb_landslide_cells += 1
                    if new_el < topo[upstream_neighbors[0]] and distance_to_crit_node < self._max_distance_of_slide_from_crit_node:
                        # Do actual slide
                        upstream_count += 1
                        height_diff = topo[upstream_neighbors[0]] - new_el
                        sed_landslide_ero = np.clip(
                            min(
                                soil_d[upstream_neighbors[0]],
                                height_diff,
                            ),
                            a_min=0.0,
                            a_max=None,
                        )
                        soil_d[upstream_neighbors[0]] -= sed_landslide_ero
                        bed_landslide_ero = np.clip(
                            bed[upstream_neighbors[0]]
                            - (new_el - soil_d[upstream_neighbors[0]]),
                            a_min=0.0,
                            a_max=None,
                        )
                        bed[upstream_neighbors[0]] -= bed_landslide_ero
                        topo[upstream_neighbors[0]] = new_el
                        
                        
                        vol_sed = (
                                sed_landslide_ero * (1 - self._phi) * (self.grid.dx ** 2)
                        )
                        vol_bed = bed_landslide_ero * (self.grid.dx ** 2)
                        store_volume_sed = store_volume_sed + vol_sed
                        store_volume_bed = store_volume_bed + vol_bed

                        neighbors = np.concatenate(
                            (
                                self.grid.active_adjacent_nodes_at_node[
                                    upstream_neighbors[0]
                                ],
                                self.grid.diagonal_adjacent_nodes_at_node[
                                    upstream_neighbors[0]
                                ],
                            )
                        )
                        neighbors = neighbors[neighbors != -1]
                        neighbors_up = neighbors[topo[neighbors] > crit_node_el]
                        upstream_neighbors = [*upstream_neighbors, *neighbors_up]

                        temp, idx = np.unique(upstream_neighbors, return_index=True)
                        upstream_neighbors = np.array(upstream_neighbors)
                        upstream_neighbors = upstream_neighbors[np.sort(idx)]
                        if not self._landslides_on_boundary_nodes:
                            upstream_neighbors = upstream_neighbors[
                                ~self.grid.node_is_boundary(upstream_neighbors)
                            ]
                        # if one of the LS pixels also appears in critical_landslide_nodes list,
                        # remove it there so that no new landslide is initialized
                        critical_landslide_nodes = critical_landslide_nodes[np.where((critical_landslide_nodes != upstream_neighbors[0]))]

                    upstream_neighbors = np.delete(upstream_neighbors, 0, 0)

            if critical_landslide_nodes.size > 0:
                critical_landslide_nodes = np.delete(critical_landslide_nodes, 0, 0)

        topo_diff = before_topo_copy[:] - topo[:]
        indices = np.argwhere((topo_diff > 0))
        if np.size(indices) > 0:
            indices = np.delete(indices,
                                self._grid.node_is_boundary(indices).flatten())

        soil_erosion_depth = np.min((before_soil_copy[indices], topo_diff[indices]), axis=0)
        store_volume_sed = soil_erosion_depth * (1 - self._phi) * (self.grid.dx ** 2)
        store_volume_bed = (topo_diff[indices] - soil_erosion_depth) * (self.grid.dx ** 2)

        landslide_sed_in_soil[indices] = store_volume_sed
        landslide_sed_in_bedrock[indices] = store_volume_bed
        soil_removal[indices] -= soil_erosion_depth

        soil_d[self.grid.core_nodes] = before_soil_copy[self.grid.core_nodes]
        soil_d[indices] -= soil_erosion_depth
        bed[self.grid.core_nodes] = topo[self.grid.core_nodes] - soil_d[self.grid.core_nodes]
        a = np.sum(grain_fractions[indices, :], axis=0)
        b = np.sum(np.sum(grain_fractions[indices, :], axis=0))
        grain_fractions_failure = np.divide(a, b, where=b != 0)
        self._soillandslide_grainsize_fractures = b * grain_fractions_failure.flatten()
        
        ## Update the sliding mass
        self._update_mass()

    def failures_runout(self):

        topo = self._grid.at_node["topographic__elevation"]
        bed = self._grid.at_node["bedrock__elevation"]
        soil_d = self._grid.at_node["soil__depth"]
        landslide_depo_soil = self._grid.at_node["landslide__deposition_soil"]
        landslide_depo_bedrock = self._grid.at_node["landslide__deposition_bedrock"]
        landslide_depo_soil.fill(0)
        landslide_depo_bedrock.fill(0)
        grain_weight = self._grid.at_node['grains__weight']
        stack_rev = np.flip(self.grid.at_node["flow__upstream_node_order"])
        threshold_slope_dzdx = self._threshold_slope
        receivers = self._grid.at_node["hill_flow__receiver_node"]
        fract_receivers = self._grid.at_node["hill_flow__receiver_proportions"]
        node_status = self._grid.status_at_node

        Qin_hill_soil = self._grid.at_node["landslide_soil_sediment_point_source"]  # m^3 for time step
        Qin_hill_bedrock = self._grid.at_node["landslide_bedrock_sediment_point_source"]  # m^3 for time step
        Qout_hill_bedrock = np.zeros_like(Qin_hill_soil)
        Qout_hill_soil = np.zeros_like(Qin_hill_soil)

        dh_hill = np.zeros_like(Qin_hill_soil)  # deposition dz
        max_D = np.zeros_like(Qin_hill_soil)
        Qout_hill = np.zeros_like(Qin_hill_soil)
        Qin_hill = Qin_hill_soil + Qin_hill_bedrock


        if np.any(Qin_hill > 0):

            slope = self._grid.at_node["hill_topographic__steepest_slope"]
            slope_min = np.min(slope, axis=1)

            slope_copy = np.copy(slope)
            slope_copy[slope_copy <= 0] = np.inf
            slope_copy_min = np.min(slope_copy, axis=1)
            slope_copy_min[slope_copy_min == np.inf] = 0

            slope = np.max((slope_min, slope_copy_min), axis=0)
            slope[slope <= 0] = 0

            topo_copy = np.array(topo.copy())
            length_adjacent_cells = np.array([self._grid.dx, self._grid.dx, self._grid.dx, self._grid.dx,
                                              self._grid.dx * np.sqrt(2), self._grid.dx * np.sqrt(2),
                                              self._grid.dx * np.sqrt(2), self._grid.dx * np.sqrt(2)])

            stack_rev_sel = stack_rev[node_status[stack_rev] == NodeStatus.CORE]
            L_Hill = np.where(
                slope < threshold_slope_dzdx,
                self._grid.dx / (1 - (slope / threshold_slope_dzdx) ** 2),
                1e6,
            )

            for i, donor in enumerate(stack_rev_sel):
                donor_elev = topo_copy[donor]

                dH = max(
                    0,
                    min(((Qin_hill[donor] / self._grid.dx) / L_Hill[donor]) / (1 - self._phi), max_D[donor])
                )


                # from here
                neighbors = np.concatenate(
                    (
                        self._grid.active_adjacent_nodes_at_node[
                            donor
                        ],
                        self._grid.diagonal_adjacent_nodes_at_node[
                            donor
                        ],
                    )
                )

                neighbors = neighbors[
                    ~self._grid.node_is_boundary(neighbors)]
                neighbors = neighbors[neighbors > 0]
                neibhors_elev = topo_copy[neighbors]

                downstream_neibhors = neighbors[neibhors_elev < donor_elev]
                downstream_neibhors_elev = topo_copy[downstream_neibhors]

                if np.size(downstream_neibhors_elev) == 0:
                    if np.size(neibhors_elev[neibhors_elev > donor_elev]) > 0:
                        max_diff_downstream = np.max(neibhors_elev[neibhors_elev > donor_elev]) - donor_elev
                    else:
                        max_diff_downstream = 0
                else:
                    # calc maximal diffrence to downstream
                    elev_diff_downstream = (donor_elev - downstream_neibhors_elev)
                    max_diff_downstream_indx = np.argmax((elev_diff_downstream))
                    max_diff_downstream_node = downstream_neibhors[max_diff_downstream_indx]

                    dist_to_downstream_neibhor = np.sqrt(
                        (self.grid.node_x[max_diff_downstream_node] - self.grid.node_x[donor]) ** 2 + (
                                self.grid.node_y[max_diff_downstream_node] - self.grid.node_y[donor]) ** 2)

                    max_diff_downstream = np.max((0, (topo_copy[
                                                          max_diff_downstream_node] + 2 * dist_to_downstream_neibhor) -
                                                  topo_copy[donor]))


                dH = np.min((dH, max_diff_downstream))

                dH_volume = (dH * self._grid.dx ** 2) * (1 - self._phi)

                Qin_ratio_soil = np.divide(Qin_hill_soil[donor], Qin_hill[donor],
                                           where=Qin_hill[donor] > 0,
                                           out=np.zeros_like(Qin_hill[donor]))
                deposited_soil_flux  = np.min((Qin_ratio_soil * dH_volume,
                                               Qin_hill_soil[donor]),
                                              axis=0)
                dH_volume -= deposited_soil_flux

                Qin_hill_soil[donor] -= deposited_soil_flux
                Qout_hill_soil[donor] += Qin_hill_soil[donor]

                Qin_hill_bedrock[donor] -= dH_volume
                Qout_hill_bedrock[donor] += Qin_hill_bedrock[donor]

                landslide_depo_soil[donor] += (deposited_soil_flux / self._grid.dx ** 2) / (1 - self._phi)
                landslide_depo_bedrock[donor] += (dH_volume / self._grid.dx ** 2) / (1 - self._phi)

                Qin_hill[donor] -= (dH_volume + deposited_soil_flux)
                Qout_hill[donor] += Qin_hill[donor]

                dh_hill[donor] += dH
                topo_copy[donor] += dH

                for r in range(receivers.shape[1]):
                    rcvr = receivers[donor, r]

                    max_D_angle = topo_copy[donor] - self._min_deposition_slope * length_adjacent_cells[r] - topo_copy[
                        rcvr]
                    max_D[rcvr] = min(max(max_D[rcvr], topo_copy[donor] - topo_copy[rcvr]), max_D_angle)

                    proportion = fract_receivers[donor, r]
                    if proportion > 0. and donor != rcvr:
                        Qin_hill[rcvr] += Qout_hill[donor] * proportion
                        Qin_hill_soil[rcvr] += Qout_hill_soil[donor] * proportion
                        Qin_hill_bedrock[rcvr] += Qout_hill_bedrock[donor] * proportion

                        Qin_hill[donor] -= Qout_hill[donor] * proportion
                        Qin_hill_soil[donor] -= Qout_hill_soil[donor] * proportion
                        Qin_hill_bedrock[donor] -= Qout_hill_bedrock[donor] * proportion


            soil_d[self._grid.core_nodes] += dh_hill[self._grid.core_nodes]
            topo[self._grid.core_nodes] = bed[self._grid.core_nodes] + soil_d[self._grid.core_nodes]
            dh_hill[:] = 0
            Qin_hill[:] = 0
            Qin_hill_soil[:] = 0
            Qin_hill_bedrock[:] = 0
            self._update_mass()

    def _update_mass(self):

        sigma = self._rho_r
        grain_weight = self._grid.at_node['grains__weight']
        landslide_depo_soil = self._grid.at_node["landslide__deposition_soil"]
        landslide_depo_bedrock = self._grid.at_node["landslide__deposition_bedrock"]
        deposition_indices_bedrock = np.where(landslide_depo_bedrock != 0)[0]
        deposition_indices_soil = np.where(landslide_depo_soil != 0)[0]

        if np.any(landslide_depo_bedrock):
            deposition_mass = (
                                      landslide_depo_bedrock[deposition_indices_bedrock] * self._grid.dx ** 2) \
                              * sigma * (1 - self._phi)  # in Kg

            for (index, mass) in zip(deposition_indices_bedrock, deposition_mass):
                # Add mass based on grain size distribution
                add_mass_vec_ratio = mass / np.sum(
                    self._bedrock_sediment_grainsizes)  
                grain_weight[index, :] += add_mass_vec_ratio * self._bedrock_sediment_grainsizes 

        if np.any(landslide_depo_soil):
            deposition_mass = (
                                      landslide_depo_soil[deposition_indices_soil] * self._grid.dx ** 2) \
                              * sigma * (1 - self._phi)  # in Kg

            for (index, mass) in zip(deposition_indices_soil, deposition_mass):

                if mass > 0:
                    add_mass_vec_ratio = mass / np.sum(self._soillandslide_grainsize_fractures)
                    grain_weight[index,
                    :] += add_mass_vec_ratio * self._soillandslide_grainsize_fractures  
                else:
                    in_fractions = np.argwhere((grain_weight[index, :] > 0)).tolist()
                    mass_by_fraction = mass * (
                                grain_weight[index, in_fractions] / np.sum(grain_weight[index, in_fractions]))
                    grain_weight[index,
                                 in_fractions] += mass_by_fraction

        landslide_depo_soil[:] = 0
        landslide_depo_bedrock[:] = 0

    def _update_diffusive_mass(self, flux):

        bed = self._grid.at_node['bedrock__elevation']
        soil = self._grid.at_node['soil__depth']
        grain_weight_node = self._grid.at_node['grains__weight']
        grain_weight_node[grain_weight_node<0] = 0
        self._sigma  = self._rho_r
        g_total_dt_node = np.sum(grain_weight_node, 1).copy()  # Total grain size mass at node
        g_total_link = self._zeros_at_link.copy()  # Total grain size mass at link
        g_state_link = self._zeros_at_link_for_fractions.copy()  # Grain size mass for each size fraction
        sed_flux_at_link = self._zeros_at_link.copy()
        self._sum_dzdt = self._zeros_at_node.copy()
        landslide_depo_soil = self._grid.at_node["landslide__deposition_soil"]
        landslide_depo_bedrock = self._grid.at_node["landslide__deposition_bedrock"]
        landslide_depo_bedrock.fill(0.)
        landslide_depo_soil.fill(0.)
        upwind_node_id_at_link  = self._grid.map_value_at_max_node_to_link('topographic__elevation', self._grid.nodes.flatten(),)
        nonzero_upwind_node_ids = np.uint(upwind_node_id_at_link[np.nonzero(upwind_node_id_at_link)])
        nonzero_downind_link_ids = np.nonzero(upwind_node_id_at_link)[0]
        tgrad = self.grid.calc_grad_at_link(self.grid.at_node['topographic__elevation'])

        g_total_link[nonzero_downind_link_ids] = g_total_dt_node[
            nonzero_upwind_node_ids]  # Total sediment mass for of each up-wind node mapped to link.
        g_state_link[nonzero_downind_link_ids, :] = grain_weight_node[nonzero_upwind_node_ids,
                                                    :]  # Fraction of sediment mass of each upwind node, for all size-fraction, mapped to link
        g_fraction_link = np.divide(g_state_link,
                                    g_total_link.reshape(-1, 1),
                                    out=np.zeros_like(g_state_link),
                                    where=g_total_link.reshape(-1, 1) != 0)



        self._sed_flux_at_link_class = np.multiply(np.abs(flux.reshape([-1, 1])),
                                                   g_fraction_link) * -np.sign( tgrad[:,np.newaxis] )


        outlinks_at_node = self.grid.link_at_node_is_downwind(tgrad)
        fluxes = self._grid.link_dirs_at_node[:, :, np.newaxis] * self._sed_flux_at_link_class[self._grid.links_at_node,
                                                                  :]
        outlinks_fluxes_at_node = np.copy(fluxes)
        outlinks_fluxes_at_node[~outlinks_at_node, :] = 0
        outlinks_id = self.grid.links_at_node[outlinks_at_node]
        inlinks_fluxes_at_node = np.copy(fluxes)
        inlinks_fluxes_at_node[outlinks_at_node, :] = 0

        # Sum all outfluxes per node to check if time step need to be reduced.
        sum_fluxes_out = np.abs(np.sum(np.abs(outlinks_fluxes_at_node), 1))
        dz_per_grainsize_at_node = np.abs(grain_weight_node / (self._sigma * (1 - self._phi) * self.grid.dx ** 2))
        indices_to_correct_flux = np.where(sum_fluxes_out / self.grid.dx > dz_per_grainsize_at_node)

        if np.any(indices_to_correct_flux):
            ratios = np.divide(dz_per_grainsize_at_node[indices_to_correct_flux],
                               sum_fluxes_out[indices_to_correct_flux])
            # If the outflux are greater than what exist in the flow, correct the outflux.
            outlinks_fluxes_at_node[indices_to_correct_flux[0], :, indices_to_correct_flux[1]] *= ratios[:,
                                                                                                  np.newaxis]
            # Update the weight flux.
            self._sed_flux_at_link_class[outlinks_id] = np.abs(outlinks_fluxes_at_node[outlinks_at_node, :]) * np.sign(self._sed_flux_at_link_class[outlinks_id])

        sed_flux_at_link[:] = np.sum(self._sed_flux_at_link_class , axis=1)
        soil_sediment_flux_at_link_fractions = self._sed_flux_at_link_class # flux only from soil sediment


        dzdt_soil_sediment = -self._grid.calc_flux_div_at_node(sed_flux_at_link)
        bedrock_sediment_flux_at_link = (np.abs(flux) - np.abs(sed_flux_at_link ))*np.sign(flux)
        dzdt_bedrock_sediment = -self._grid.calc_flux_div_at_node(bedrock_sediment_flux_at_link)


        # Update weights according to diffusive soil sediment
        for size_class in range(np.shape(soil_sediment_flux_at_link_fractions)[1]):
            dzdt = -self._grid.calc_flux_div_at_node(soil_sediment_flux_at_link_fractions[:, size_class])
            grain_weight_node[:, size_class] += (dzdt * self._grid.dx ** 2) * self._rho_r * (1 - self._phi)  # in kg
        grain_weight_node[grain_weight_node<0]=0

        ## Bedrock sediment section
        erosion_bedrock_indices = np.where(dzdt_bedrock_sediment < 0)
        deposition_indices = np.where(dzdt_bedrock_sediment >= 0)
        landslide_depo_bedrock[deposition_indices] += dzdt_bedrock_sediment[deposition_indices]

        # Update topography (bedrock and soil depth)
        bed[erosion_bedrock_indices] += dzdt_bedrock_sediment[erosion_bedrock_indices] 
        soil +=  dzdt_soil_sediment
        soil[deposition_indices] +=  dzdt_bedrock_sediment[deposition_indices]
        soil[soil<=0]=0
        self._update_mass()

        
    def run_one_step(self):
        
        self.failures_erosion()
        self.failures_runout()
        return 

