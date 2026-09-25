import numpy as np
cimport cython
cimport numpy as cnp
from cython.parallel cimport prange
from libc.stdio cimport printf
from libc.math cimport log
from cython.parallel cimport prange
from cpython cimport bool as bool
from cython.parallel cimport parallel
cimport openmp
import numpy as np
cimport cython
from libc.math cimport pow
from libc.math cimport INFINITY, fabs

# cdef int num_threads
#
# openmp.omp_set_dynamic(1)
# with nogil, parallel():
#     num_threads = openmp.omp_get_num_threads()
#     # ...
from libc.stdio cimport printf

#include <math.h>
ctypedef fused id_t:
    cython.integral
    long long

ctypedef fused float_or_int:
    cython.integral
    cython.floating


DTYPE_INT = np.intc
ctypedef cnp.int64_t DTYPE_INT_t

DTYPE_FLOAT = np.double
DTYPE_complex = np.complexfloating
ctypedef cnp.double_t DTYPE_FLOAT_t

ctypedef cnp.uint8_t uint8

# Thread count for all prange(...) loops below. This used to be hardcoded to
# 32, which oversubscribes almost any machine (these loops run over ~n_nodes
# or ~n_links, called 1000+ times per storm, so 32-thread team setup/teardown
# overhead on every call adds up fast). Sizing this to the actual core count
# instead measured ~18% faster end-to-end in profiling, with identical
# output. Falls back to 4 if the core count can't be determined.
import os as _os
cdef int N_THREADS = _os.cpu_count() or 4



@cython.boundscheck(False)
@cython.wraparound(False)
def grain_size_sum_at_node(
        cython.floating[:, :] value_at_node_per_size,
        cython.floating[:] out,
        shape
):
    cdef int n_nodes = shape[0]
    cdef int n_cols = shape[1]
    cdef int col, node


    for node in prange(n_nodes, nogil=True, schedule="static",num_threads=N_THREADS):
        for col in range(n_cols):
            out[node]  = out[node] + value_at_node_per_size[node, col]

    return out.base



@cython.boundscheck(False)
@cython.wraparound(False)
def calc_concentration(
    cython.floating[:, :] out,
    const cython.floating[:, :] value_at_node_per_size,
    const cython.floating[:] value_at_node,
    shape,
):
    cdef int n_nodes = shape[0]
    cdef int n_cols = shape[1]

    cdef int index, col, gs, node
    cdef int link

    for node in prange(n_nodes, nogil=True, schedule="static",num_threads=N_THREADS):
        for col in range(n_cols):
            out[node, col] = value_at_node_per_size[node, col] / value_at_node[node]

    return out.base



def sum_out_discharge(
        cnp.ndarray[DTYPE_INT_t, ndim=1] upwind_node_at_link,
        cnp.ndarray[DTYPE_FLOAT_t, ndim=1] abs_discharge,
        cnp.ndarray[DTYPE_INT_t, ndim=1] link_list,
        cnp.ndarray[DTYPE_FLOAT_t, ndim=1] out_discharge_at_node,
        shape,
        ):
        """Scatter-accumulate link discharge onto each link's upwind node.

        Deliberately SERIAL, not prange. Multiple links routinely share the
        same upwind node (any interior node has several links touching it),
        so this is a scatter-add into a node-indexed array from a
        link-indexed loop: `out_discharge_at_node[upwind_node] += ...`. Under
        prange, two threads can race to read-modify-write the same
        out_discharge_at_node[upwind_node] slot at once with no
        synchronization, silently dropping one thread's contribution. This
        was verified empirically (a 30-trial randomized test against a
        serial reference reproduced wrong, non-deterministic results in
        about half the trials on just 2 threads) - it's the kind of bug that
        surfaces as spatially patchy, run-to-run-inconsistent discharge/
        erosion output, which is worse than the parallel speedup is worth
        for a loop this cheap (one add per link).
        """

        cdef int index, link, upwind_node
        cdef int n_links = shape

        for index in range(n_links):
            link = link_list[index]
            upwind_node = upwind_node_at_link[link]
            out_discharge_at_node[upwind_node] += abs_discharge[link]

        return out_discharge_at_node


def calc_flux_at_link(
        const double dx,
        const double sigma,
        const double phi,
        cython.floating[:, : ] weight_flux_at_link,
        cython.floating[:] water_surface_grad_at_link,
        cython.floating[:, :] sediments_flux_at_link,
        shape,
):
    cdef int n_links = shape[0]
    cdef int n_cols = shape[1]
    cdef int col, link, index


    for link in prange(n_links, nogil=True, schedule="static", num_threads=N_THREADS):
        for col in range(n_cols):
            sediments_flux_at_link[link, col]  = water_surface_grad_at_link[link] * (weight_flux_at_link[link, col] / (dx * sigma * (1 - phi)))

    return sediments_flux_at_link.base




def get_outin_fluxes(
        cnp.ndarray[DTYPE_INT_t, ndim=1] upwind_node_at_link,
        cnp.ndarray[DTYPE_INT_t, ndim=1] downwind_node_at_link,
        cnp.ndarray[DTYPE_FLOAT_t, ndim=2] weight_flux_at_link,
        cnp.ndarray[DTYPE_INT_t, ndim=1] link_list,
        cnp.ndarray[DTYPE_FLOAT_t, ndim=2] outlinks_fluxes_at_node,
        cnp.ndarray[DTYPE_FLOAT_t, ndim=2] inlinks_fluxes_at_node,
        cnp.ndarray[DTYPE_FLOAT_t, ndim=1] total_outflux_at_node,
        cnp.ndarray[DTYPE_FLOAT_t, ndim=1] total_influx_at_node,
        shape,
        ):

        cdef int node, index, index_inlink, index_outlink, l_inlink, l_outlink, gs, link, upwind_node, downwind_node
        cdef int n_links = shape[0]
        cdef int n_gs  = shape[1]

        # Deliberately SERIAL, not prange - same reasoning as sum_out_discharge
        # above. This scatter-accumulates per-link flux onto each link's
        # upwind/downwind node (`inlinks_fluxes_at_node[downwind_node, gs] +=
        # ...` etc.); since multiple links commonly share a node, running this
        # over prange races multiple threads on the same node's accumulator
        # with no synchronization, silently losing updates.
        for index in range(n_links):
            link = link_list[index]

            upwind_node = upwind_node_at_link[link]
            downwind_node = downwind_node_at_link[link]

            for gs in range(n_gs):
                inlinks_fluxes_at_node[downwind_node, gs] += weight_flux_at_link[link, gs]
                outlinks_fluxes_at_node[upwind_node, gs] += weight_flux_at_link[link, gs]
                total_outflux_at_node[upwind_node] += weight_flux_at_link[link, gs]
                total_influx_at_node[downwind_node] += weight_flux_at_link[link, gs]

        return outlinks_fluxes_at_node, inlinks_fluxes_at_node, total_outflux_at_node, total_influx_at_node



def calc_CQ(
        cnp.ndarray[DTYPE_FLOAT_t, ndim=2] c_kg,
        cnp.ndarray[DTYPE_FLOAT_t, ndim=2] CQ,
        cnp.ndarray[DTYPE_FLOAT_t, ndim=1] q,
        cnp.ndarray[DTYPE_INT_t, ndim=1] core_nodes,
        shape,
        grid_dx,
        ):

        cdef int node, index, gs
        cdef int dx = grid_dx
        cdef int n_nodes = shape[0]
        cdef int n_gs = shape[1]

        for index in prange(n_nodes, nogil=True, schedule="static", num_threads=N_THREADS):
            node = core_nodes[index]

            for gs in range(n_gs):
                CQ[node, gs] = c_kg[node, gs] * q[node] * dx

        return CQ



def calc_flux_at_link_per_size(
        cnp.ndarray[DTYPE_FLOAT_t, ndim=1] q_water_at_link,
        cnp.ndarray[DTYPE_FLOAT_t, ndim=2] suspended__sediments_concentration_at_link,
        cnp.ndarray[DTYPE_INT_t, ndim=1] active_links,
        cnp.ndarray[DTYPE_FLOAT_t, ndim=2] weight_flux_at_link,
        shape
        ):

        cdef int link, index, gs
        cdef int n_links = shape[0]
        cdef int n_gs = shape[1]

        for index in prange(n_links, nogil=True, schedule="static", num_threads=N_THREADS):
            link = active_links[index]

            for gs in range(n_gs):
                weight_flux_at_link[link, gs] = q_water_at_link[link] * suspended__sediments_concentration_at_link[link, gs]

        return weight_flux_at_link



def calc_DR(
        cnp.ndarray[DTYPE_FLOAT_t, ndim=1] flow_width,
        cnp.ndarray[DTYPE_FLOAT_t, ndim=2] CQ,
        cnp.ndarray[DTYPE_FLOAT_t, ndim=2] TC,
        cnp.ndarray[DTYPE_FLOAT_t, ndim=2] Dc,
        cnp.ndarray[DTYPE_FLOAT_t, ndim=1] vs,
        cnp.ndarray[DTYPE_INT_t, ndim=1] active_nodes,
        cnp.ndarray[DTYPE_FLOAT_t, ndim=1] q_at_node,
        cnp.ndarray[DTYPE_FLOAT_t, ndim=2] out,
        dx_c,
        shape
):


    cdef int n_nodes = shape[0]
    cdef int n_cols = shape[1]
    cdef int dx = dx_c
    cdef int  col, node, index
    cdef double b
    cdef double condition

    for index in prange(n_nodes, nogil=True, schedule="static", num_threads=N_THREADS):
        node = active_nodes[index]
        for col in range(n_cols):
            condition = TC[node, col] * flow_width[node]
            if CQ[node, col] < condition:
                out[node, col] = ((1 - (CQ[node, col] / condition)) * Dc[node,col]) /  dx
            elif CQ[node, col] > condition:
                b = condition - CQ[node, col]
                out[node,col] = (b * (0.5 * vs[col]) / q_at_node[node]) / dx
            # else:
            #     out[node,col] = 0.00000001

    return out



def calc_Dc(
        cnp.ndarray[DTYPE_FLOAT_t, ndim=1] tau_s,
        cnp.ndarray[DTYPE_FLOAT_t, ndim=2] tau_c,
        cnp.ndarray[DTYPE_INT_t, ndim=1] core_nodes,
        cnp.ndarray[DTYPE_FLOAT_t, ndim=2] out,
        kr_c,
        shape,
        ):
        # Dc = kr * (tau_s - tau_c)  ->  units kg/(m^2 s)
        # kr expected in [s/m] (standard WEPP rill erodibility convention),
        # tau in [Pa] = kg/(m s^2). Kept as the reference/default detachment model.

        cdef int n_nodes = shape[0]
        cdef int n_gs = shape[1]
        cdef double kr = kr_c
        cdef double excess_stress
        cdef int node, index, gs

        for index in prange(n_nodes, nogil=True, schedule="static", num_threads=N_THREADS):
            node = core_nodes[index]

            for gs in range(n_gs):
                excess_stress = tau_s[node] - tau_c[node, gs]
                if excess_stress>0:
                    out[node, gs] = excess_stress * kr

        return out


@cython.boundscheck(False)
@cython.wraparound(False)
def calc_Dc_stream_power(
        cnp.ndarray[DTYPE_FLOAT_t, ndim=1] stream_power,
        cnp.ndarray[DTYPE_INT_t, ndim=1] core_nodes,
        cnp.ndarray[DTYPE_FLOAT_t, ndim=2] out,
        k_omega_c,
        shape,
        ):
        # RHEM V2.3 concentrated-flow detachment (Al-Hamdan et al., 2012b):
        #   Dc = K_omega * omega,  omega = rho * g * S * q  (stream power, kg/s^3)
        # No critical-shear threshold: detachment starts as soon as
        # concentrated flow starts. `stream_power` (= rho*g*S*q per node) is
        # precomputed by the caller and passed in directly.
        #
        # Units, matched to calc_Dc's kg/(m^2 s) output:
        #   q (unit-width discharge) in [m^2/s]  ->  omega in [kg/s^3]
        #   k_omega in [s^2/m^2]  (same convention as RHEM's K_omega)
        #   -> Dc = k_omega * omega  has units kg/(m^2 s), matching calc_Dc.

        cdef int n_nodes = shape[0]
        cdef int n_gs = shape[1]
        cdef double k_omega = k_omega_c
        cdef double dc_node
        cdef int node, index, gs

        for index in prange(n_nodes, nogil=True, schedule="static", num_threads=N_THREADS):
            node = core_nodes[index]

            dc_node = k_omega * stream_power[node]
            if dc_node < 0:
                dc_node = 0.0

            for gs in range(n_gs):
                out[node, gs] = dc_node

        return out


def calc_detached_deposited(
    cnp.ndarray[DTYPE_FLOAT_t, ndim = 2] DR,
    cnp.ndarray[DTYPE_FLOAT_t, ndim = 2] DR_abs,
    cnp.ndarray[DTYPE_FLOAT_t, ndim = 2] grain_weight_at_node,
    cnp.ndarray[DTYPE_FLOAT_t, ndim = 2] grain_fractions_at_node,
    cnp.ndarray[DTYPE_FLOAT_t, ndim = 2] deatched_soil_weight,
    cnp.ndarray[DTYPE_FLOAT_t, ndim = 2] deatched_bedrock_weight,
    cnp.ndarray[DTYPE_FLOAT_t, ndim = 2] suspended_fraction_at_node,
    cnp.ndarray[DTYPE_FLOAT_t, ndim = 2] bedrock_grain_fractions,
    cnp.ndarray[DTYPE_FLOAT_t, ndim = 2] temp_suspended_sediment_weight_at_node_per_size,
    cnp.ndarray[DTYPE_FLOAT_t, ndim = 2] deposited_suspended_sediments_weights_at_node,
    cnp.ndarray[DTYPE_FLOAT_t, ndim = 1] total_deposited_sediments_dz_at_node,
    cnp.ndarray[DTYPE_FLOAT_t, ndim = 1] entrainment_soil_rate_dz,
    cnp.ndarray[DTYPE_FLOAT_t, ndim = 1] entrainment_bedrock_rate_dz,
    cnp.ndarray[DTYPE_FLOAT_t, ndim = 1] soil_e_expo,
    cnp.ndarray[DTYPE_INT_t, ndim = 1] core_nodes,
    factor_convert_weight_to_dz_c,
    factor_convert_weight_to_dz_bedrock_c,
    shape,
    dx_c
    ):

    cdef int n_nodes = shape[0]
    cdef int n_gs = shape[1]
    cdef int node, index, gs
    cdef double dr_node_per_gs, detached_soil_weight_at_node,\
        deatched_bedrock_weight_at_node, deposited_weight, summed_deposited_at_node,\
        summed_detached_soil_weight_at_node, summed_detached_bedrock_weight_at_node
    cdef double factor_convert_weight_to_dz = factor_convert_weight_to_dz_c
    cdef double factor_convert_weight_to_dz_bedrock = factor_convert_weight_to_dz_bedrock_c

    cdef double dx = dx_c


    for index in prange(n_nodes, nogil=True, schedule="static", num_threads=N_THREADS):
        node = core_nodes[index]
        summed_deposited_at_node = 0
        summed_detached_soil_weight_at_node = 0
        summed_detached_bedrock_weight_at_node = 0

        for gs in range(n_gs):
            dr_node_per_gs = DR[node, gs]

            if dr_node_per_gs > 0:

                ## Detached soil weight
                detached_soil_weight_at_node =  dr_node_per_gs  * soil_e_expo[node]
                detached_soil_weight_at_node = detached_soil_weight_at_node * grain_fractions_at_node[node, gs]

                if detached_soil_weight_at_node > grain_weight_at_node[node,gs]:
                    detached_soil_weight_at_node = grain_weight_at_node[node,gs]
                deatched_soil_weight[node, gs] = detached_soil_weight_at_node

                summed_detached_soil_weight_at_node = summed_detached_soil_weight_at_node + detached_soil_weight_at_node


                ## Detached bedrock weight
                deatched_bedrock_weight_at_node = dr_node_per_gs  * (1 - soil_e_expo[node])
                deatched_bedrock_weight[node, gs] = deatched_bedrock_weight_at_node * bedrock_grain_fractions[node,gs]
                summed_detached_bedrock_weight_at_node = summed_detached_bedrock_weight_at_node + deatched_bedrock_weight[node, gs] #deatched_bedrock_weight_at_node

                ## add to suspended
                temp_suspended_sediment_weight_at_node_per_size[node, gs] += detached_soil_weight_at_node + deatched_bedrock_weight_at_node

            if dr_node_per_gs < 0:
                dr_node_per_gs = DR_abs[node, gs]
                deposited_weight = dr_node_per_gs * dx * dx * suspended_fraction_at_node[node, gs]

                # Clamp to what's actually available. Written as `not (<=)`
                # rather than `>` so it also catches NaN/inf (e.g. 0*inf):
                # NaN fails both `<=` and `>`, so a plain `>` check would
                # silently let NaN through uncapped.
                if not (deposited_weight <= temp_suspended_sediment_weight_at_node_per_size[node, gs]):
                     deposited_weight = temp_suspended_sediment_weight_at_node_per_size[node, gs] #*50

                deposited_suspended_sediments_weights_at_node[node, gs] = deposited_weight
                summed_deposited_at_node  = summed_deposited_at_node + deposited_weight

        entrainment_soil_rate_dz[node] = summed_detached_soil_weight_at_node / factor_convert_weight_to_dz
        entrainment_bedrock_rate_dz[node] = summed_detached_bedrock_weight_at_node / factor_convert_weight_to_dz_bedrock

        total_deposited_sediments_dz_at_node[node] = summed_deposited_at_node / factor_convert_weight_to_dz


    return (deatched_soil_weight,
            deatched_bedrock_weight,
            entrainment_soil_rate_dz,
            entrainment_bedrock_rate_dz,
            deposited_suspended_sediments_weights_at_node,
            total_deposited_sediments_dz_at_node)




@cython.boundscheck(False)
@cython.wraparound(False)
def calc_TC(
        const double alpha,
        const double beta,
        cython.floating[:] median_sizes,
        cython.floating[:, :] fraction_sizes,
        cython.floating[:] tau_s,
        cython.floating[:, :] TC,
        sg_c,
        rho_c,
        cnp.ndarray[DTYPE_INT_t, ndim = 1] core_nodes,
        const_sg_g_rho,
        shape
):
    cdef int n_nodes = shape[0]
    cdef int n_cols = shape[1]
    cdef int index, col, node
    cdef double y, yc, l, c, out_solv, loged_beta_plus_one
    cdef double const = 2.45
    cdef double const_b = 0.635
    cdef double sg = sg_c
    cdef double rho = rho_c
    cdef double const_c = const_sg_g_rho


    for index in prange(n_nodes, nogil=True, schedule="static", num_threads=N_THREADS):
        node = core_nodes[index]
        for col in range(n_cols):

            y = tau_s[node] / (const_c * fraction_sizes[node,col])
            yc = alpha * (fraction_sizes[node,col] /  median_sizes[node])**beta

            if y > yc:
                l = (y / yc) - 1
                c = const * sg**(-0.4) * yc**(0.5) * l
                loged_beta_plus_one  = log(c + 1)
                TC[node, col] = const_b * sg * fraction_sizes[node, col] * ((rho * tau_s[node]) ** 0.5) * l * (
                        1 -
                        ((1 / c) * loged_beta_plus_one))

    return TC.base


@cython.boundscheck(False)
@cython.wraparound(False)
def calc_TC_EH_with_discharge(
        cython.floating[:, :] fraction_sizes,
        cython.floating[:] tau_s,
        cython.floating[:] q_unit,  # Added: Unit discharge vector [m^2/s] (Q / width)
        cython.floating[:] depth,  # surface_water__depth_at_node: measured water depth [m]
        cython.floating[:, :] TC,
        sg_c,
        rho_c,
        cnp.ndarray[DTYPE_INT_t, ndim = 1] core_nodes,
        const_sg_g_rho,
        shape
):
    cdef int n_nodes = shape[0]
    cdef int n_cols = shape[1]
    cdef int index, col, node

    # Typed constants
    cdef double sg = sg_c
    cdef double rho = rho_c  # Fluid density (rho_w)
    cdef double rho_s = sg * rho  # Sediment density
    cdef double g = 9.81
    cdef double R = sg - 1.0  # Submerged specific gravity

    # Pre-calculated variables for the loop
    cdef double velocity = 0.0
    cdef double numerator_shared = 0.0
    cdef double denominator_base = pow(rho, 1.5) * pow(g, 2.0) * pow(R, 2.0)

    for index in prange(n_nodes, nogil=True, schedule="static", num_threads=N_THREADS):
        node = core_nodes[index]

        # Guard clause: If the node is dry or has zero shear stress, capacity is zero
        if tau_s[node] <= 1e-5 or q_unit[node] <= 0.0 or depth[node] <= 0.0:
            for col in range(n_cols):
                TC[node, col] = 0.0
            continue

        # Depth-averaged velocity from the field-measured water depth:
        # V = q / H, using H = surface_water__depth_at_node directly
        velocity = q_unit[node] / depth[node]

        # Pre-calculate the shared numerator for all grain sizes at this specific node
        # Engelund-Hansen: g_s ~ V^2 * tau^1.5 (velocity enters squared)
        numerator_shared = 0.05 * rho_s * pow(velocity, 2.0) * pow(tau_s[node], 1.5)

        for col in range(n_cols):
            # Dimensional Engelund-Hansen mapping per grain fraction:
            TC[node, col] = numerator_shared / (denominator_base * fraction_sizes[node, col])

    return TC.base

@cython.boundscheck(False)
@cython.wraparound(False)
def calc_flux_div_at_node(
    shape,
    xy_spacing,
    const float_or_int[:] value_at_link,
    cnp.ndarray[DTYPE_FLOAT_t, ndim=1] out , #cython.floating[:] out
):

    cdef int n_rows = shape[0]
    cdef int n_cols = shape[1]
    cdef double dx = xy_spacing[0]
    cdef double dy = xy_spacing[1]
    cdef int links_per_row = 2 * n_cols - 1
    cdef double inv_area_of_cell = 1.0 / (dx * dy)
    cdef int row, col
    cdef int node, link


    for row in prange(1, n_rows - 1, nogil=True, schedule="static"):
        node = row * n_cols
        link = row * links_per_row


        for col in range(1, n_cols - 1):
            out[node + col] = (
                dy * (value_at_link[link + 1] - value_at_link[link])
                + dx * (value_at_link[link + n_cols] - value_at_link[link - n_cols + 1])
            ) * inv_area_of_cell
            link = link + 1

    return out


@cython.boundscheck(False)
@cython.wraparound(False)
def calc_stable_dt(
    cnp.ndarray[DTYPE_FLOAT_t, ndim=1] max_downwind_gradient,
    cnp.ndarray[DTYPE_FLOAT_t, ndim=1] max_upwind_gradient,
    cnp.ndarray[DTYPE_FLOAT_t, ndim=1] detached_bedrock_dz,
    cnp.ndarray[DTYPE_FLOAT_t, ndim=1] detached_soil_dz,
    cnp.ndarray[DTYPE_FLOAT_t, ndim=1] deposited_dz,
    cnp.ndarray[DTYPE_FLOAT_t, ndim=2] suspended_weight_at_node,
    cnp.ndarray[DTYPE_FLOAT_t, ndim=2] deposited_weight,
    cnp.ndarray[DTYPE_FLOAT_t, ndim=2] suspended_dzdt_at_node,
    cnp.ndarray[DTYPE_FLOAT_t, ndim=2] outflux_weights_at_node,
    dx_c,
    dx_squared_c,
    sediment_density_c,
    porosity_c,
    max_deposition_slope_c,
    min_suspended_mass_c,
    shape,
):
    """Serial (deliberately NOT prange) reduction over the four CFL-like
    stability conditions used by `_calculate_stable_timestep` in
    OverlandflowErosionDeposition_clean.py.

    This replaces ~9 separate NumPy calls (each allocating a full-size
    temporary array) with a single pass over the nodes/grain-size classes
    that keeps everything in scalar registers. Unlike the other loops in
    this file, this one is intentionally left serial: it's a running-minimum
    reduction with a shared accumulator on every iteration, which is a poor
    fit for prange (would need a reduction clause per output, and the loop
    body is cheap/branchy rather than arithmetic-heavy) - so parallelizing it
    would likely reproduce the same overhead problem the N_THREADS note above
    describes, for little benefit since this runs once per substep rather
    than being itself a hot inner loop.
    """
    cdef int n_nodes = shape[0]
    cdef int n_gs = shape[1]
    cdef int node, gs
    cdef double dx = dx_c
    cdef double dx_squared = dx_squared_c
    cdef double sediment_density = sediment_density_c
    cdef double porosity = porosity_c
    cdef double max_deposition_slope = max_deposition_slope_c
    cdef double min_suspended_mass = min_suspended_mass_c
    cdef double dx_half = dx / 2.0
    cdef double max_dep_slope_dx = max_deposition_slope * dx
    cdef double dt_erosion = INFINITY
    cdef double dt_deposition_mass = INFINITY
    cdef double dt_deposition_topo = INFINITY
    cdef double dt_mass = INFINITY
    cdef double stable_erosion_depth, net_erosion, cand
    cdef double stable_deposition_depth, net_deposition
    cdef double total_weight_flux, swp_clamped, swp_raw, dep, outflux
    cdef bint nodes_losing_mass
    cdef bint found_deposition_mass = False
    cdef bint found_mass_loss = False

    with nogil:
        for node in range(n_nodes):
            # Condition 1: stable erosion depth
            stable_erosion_depth = max_downwind_gradient[node] * dx_half
            if stable_erosion_depth <= 0:
                stable_erosion_depth = INFINITY
            net_erosion = detached_bedrock_dz[node] + detached_soil_dz[node] - deposited_dz[node]
            if net_erosion > 0:
                cand = stable_erosion_depth / net_erosion
                if cand < dt_erosion:
                    dt_erosion = cand

            # Condition 3: stable deposition depth
            stable_deposition_depth = fabs(max_upwind_gradient[node]) * dx_half
            if stable_deposition_depth < max_dep_slope_dx:
                stable_deposition_depth = max_dep_slope_dx
            net_deposition = deposited_dz[node] - (detached_bedrock_dz[node] + detached_soil_dz[node])
            if net_deposition > 0.001:
                cand = stable_deposition_depth / net_deposition
                if cand < dt_deposition_topo:
                    dt_deposition_topo = cand

            # Condition 4 pre-check: total suspended-mass flux change at this node
            total_weight_flux = 0.0
            for gs in range(n_gs):
                total_weight_flux += (
                    suspended_dzdt_at_node[node, gs] * dx_squared *
                    sediment_density * (1.0 - porosity)
                )
            nodes_losing_mass = total_weight_flux < -min_suspended_mass
            if nodes_losing_mass:
                found_mass_loss = True

            # Conditions 2 & 4: per grain-size class
            for gs in range(n_gs):
                swp_raw = suspended_weight_at_node[node, gs]
                swp_clamped = swp_raw if swp_raw > 0 else 0.0

                dep = deposited_weight[node, gs]
                if dep > 1e-10:
                    cand = swp_clamped / dep
                    if cand < dt_deposition_mass:
                        dt_deposition_mass = cand
                    found_deposition_mass = True

                if nodes_losing_mass:
                    outflux = outflux_weights_at_node[node, gs]
                    if outflux < 0:
                        outflux = -outflux
                    if outflux > min_suspended_mass:
                        cand = swp_raw / outflux
                        if cand < dt_mass:
                            dt_mass = cand

    if found_deposition_mass:
        if dt_deposition_mass < 1.0:
            dt_deposition_mass = 1.0

    if found_mass_loss:
        if dt_mass == 0.0:
            dt_mass = INFINITY
    # If found_mass_loss is False, dt_mass stays INFINITY, matching the
    # `else: dt_mass = np.inf` branch of the original NumPy implementation.

    return dt_erosion, dt_deposition_mass, dt_deposition_topo, dt_mass

    return out
