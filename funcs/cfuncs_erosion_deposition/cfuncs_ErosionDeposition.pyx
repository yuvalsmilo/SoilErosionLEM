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

cdef int num_threads

openmp.omp_set_dynamic(1)
with nogil, parallel():
    num_threads = openmp.omp_get_num_threads()
    # ...
from libc.stdio cimport printf

#include <math.h>
ctypedef fused id_t:
    cython.integral
    long long

ctypedef fused float_or_int:
    cython.integral
    cython.floating


DTYPE_INT = np.intc
#ctypedef cnp.int_t DTYPE_INT_t
ctypedef cnp.intp_t DTYPE_INT_t
DTYPE_FLOAT = np.double
DTYPE_complex = np.complexfloating
ctypedef cnp.double_t DTYPE_FLOAT_t

ctypedef cnp.uint8_t uint8



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


    for node in prange(n_nodes, nogil=True, schedule="static",num_threads=32):
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

    for node in prange(n_nodes, nogil=True, schedule="static",num_threads=32):
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

        cdef int index, link, upwind_node
        cdef int n_links = shape

        for index in prange(n_links, nogil=True, schedule="static", num_threads=32):
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


    for link in prange(n_links, nogil=True, schedule="static", num_threads=32):
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

        for index in prange(n_links, nogil=True, schedule="static", num_threads=32):
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

        for index in prange(n_nodes, nogil=True, schedule="static", num_threads=32):
            node = core_nodes[index]

            for gs in range(n_gs):
                CQ[node, gs] = c_kg[node, gs] * q[node] * dx

        return CQ



def calc_flux_at_link_per_size(
        cnp.ndarray[DTYPE_FLOAT_t, ndim=1] q_water_at_link,
        cnp.ndarray[DTYPE_FLOAT_t, ndim=2] sediment_load_concentration_at_link,
        cnp.ndarray[DTYPE_INT_t, ndim=1] active_links,
        cnp.ndarray[DTYPE_FLOAT_t, ndim=2] weight_flux_at_link,
        shape
        ):

        cdef int link, index, gs
        cdef int n_links = shape[0]
        cdef int n_gs = shape[1]

        for index in prange(n_links, nogil=True, schedule="static", num_threads=32):
            link = active_links[index]

            for gs in range(n_gs):
                weight_flux_at_link[link, gs] = q_water_at_link[link] * sediment_load_concentration_at_link[link, gs]

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

    for index in prange(n_nodes, nogil=True, schedule="static", num_threads=32):
        node = active_nodes[index]
        for col in range(n_cols):
            condition = TC[node, col] * flow_width[node]
            if CQ[node, col] < condition:
                out[node, col] = ((1 - (CQ[node, col] / condition)) * Dc[node,col]) /  dx
            elif CQ[node, col] > condition:
                b = condition - CQ[node, col]
                out[node,col] = (b * (0.5 * vs[col]) / q_at_node[node]) / dx

    return out



def calc_Dc(
        cnp.ndarray[DTYPE_FLOAT_t, ndim=1] tau_s,
        cnp.ndarray[DTYPE_FLOAT_t, ndim=2] tau_c,
        cnp.ndarray[DTYPE_INT_t, ndim=1] core_nodes,
        cnp.ndarray[DTYPE_FLOAT_t, ndim=2] out,
        kr_c,
        shape,
        ):

        cdef int n_nodes = shape[0]
        cdef int n_gs = shape[1]
        cdef double kr = kr_c
        cdef double excess_stress
        cdef int node, index, gs

        for index in prange(n_nodes, nogil=True, schedule="static", num_threads=32):
            node = core_nodes[index]

            for gs in range(n_gs):
                excess_stress = tau_s[node] - tau_c[node, gs]
                if excess_stress>0:
                    out[node, gs] = excess_stress * kr

        return out



def calc_detached_deposited(
    cnp.ndarray[DTYPE_FLOAT_t, ndim = 2] DR,
    cnp.ndarray[DTYPE_FLOAT_t, ndim = 2] DR_abs,
    cnp.ndarray[DTYPE_FLOAT_t, ndim = 2] grain_weight_at_node,
    cnp.ndarray[DTYPE_FLOAT_t, ndim = 2] grain_fractions_at_node,
    cnp.ndarray[DTYPE_FLOAT_t, ndim = 2] deatched_soil_weight,
    cnp.ndarray[DTYPE_FLOAT_t, ndim = 2] deatched_bedrock_weight,
    cnp.ndarray[DTYPE_FLOAT_t, ndim = 2] sediment_load_fraction_at_node,
    cnp.ndarray[DTYPE_FLOAT_t, ndim = 1] bedrock_grain_fractions,
    cnp.ndarray[DTYPE_FLOAT_t, ndim = 2] temp_sediment_load_weight_at_node_per_size,
    cnp.ndarray[DTYPE_FLOAT_t, ndim = 2] deposited_sediment_load_weights_at_node,
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


    for index in prange(n_nodes, nogil=True, schedule="static", num_threads=32):
        node = core_nodes[index]
        summed_deposited_at_node = 0
        summed_detached_soil_weight_at_node = 0
        summed_detached_bedrock_weight_at_node = 0

        for gs in range(n_gs):
            dr_node_per_gs = DR[node, gs]

            if dr_node_per_gs > 0:

                ## Detached soil weight
                detached_soil_weight_at_node =  dr_node_per_gs * dx * dx  * soil_e_expo[node]
                detached_soil_weight_at_node = detached_soil_weight_at_node * grain_fractions_at_node[node, gs]

                if detached_soil_weight_at_node > grain_weight_at_node[node,gs]:
                    detached_soil_weight_at_node = grain_weight_at_node[node,gs]
                deatched_soil_weight[node, gs] = detached_soil_weight_at_node

                summed_detached_soil_weight_at_node = summed_detached_soil_weight_at_node + detached_soil_weight_at_node


                ## Detached bedrock weight
                deatched_bedrock_weight_at_node = dr_node_per_gs * dx * dx  * (1 - soil_e_expo[node])
                deatched_bedrock_weight[node, gs] = deatched_bedrock_weight_at_node * bedrock_grain_fractions[gs]
                summed_detached_bedrock_weight_at_node = summed_detached_bedrock_weight_at_node + deatched_bedrock_weight_at_node

                ## add to load
                temp_sediment_load_weight_at_node_per_size[node, gs] += detached_soil_weight_at_node + deatched_bedrock_weight_at_node

            if dr_node_per_gs < 0:
                dr_node_per_gs = DR_abs[node, gs]
                deposited_weight = dr_node_per_gs * dx * dx * sediment_load_fraction_at_node[node, gs]

                if deposited_weight > temp_sediment_load_weight_at_node_per_size[node, gs]:
                    deposited_weight = temp_sediment_load_weight_at_node_per_size[node, gs]

                deposited_sediment_load_weights_at_node[node, gs] = deposited_weight
                summed_deposited_at_node  = summed_deposited_at_node + deposited_weight

        entrainment_soil_rate_dz[node] = summed_detached_soil_weight_at_node / factor_convert_weight_to_dz
        entrainment_bedrock_rate_dz[node] = summed_detached_bedrock_weight_at_node / factor_convert_weight_to_dz_bedrock

        total_deposited_sediments_dz_at_node[node] = summed_deposited_at_node / factor_convert_weight_to_dz


    return (deatched_soil_weight,
            deatched_bedrock_weight,
            entrainment_soil_rate_dz,
            entrainment_bedrock_rate_dz,
            deposited_sediment_load_weights_at_node,
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


    for index in prange(n_nodes, nogil=True, schedule="static", num_threads=32):
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
