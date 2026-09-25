"""Landlab component that simulates fragmentation of soil grains through time.

Examples
--------
>>> import numpy as np
>>> from landlab import RasterModelGrid
>>> from landlab.components.soil_grading import SoilGrading

Create a grid on which to simulate soil grading.
>>> mg = RasterModelGrid((4, 5))

Create topographic_elevation and bedrock__elevation fields
>>> _ = mg.add_zeros("topographic__elevation", at="node")
>>> _ = mg.add_zeros("bedrock__elevation", at="node")

Initialise the SoilGrading component
>>> soil_grading = SoilGrading(mg)

Soil grading component created a soil layer:
>>> mg.at_node["soil__depth"][mg.core_nodes]
array([2., 2., 2., 2., 2., 2.])

Run one step of soil fragmentation
>>> soil_grading.run_one_step()
>>> mg.at_node["median_size__mass"][mg.core_nodes]
array([0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001])
"""

import random
import warnings

import numpy as np

from landlab import Component


class SoilGrading(Component):
    """Simulate fragmentation of soil grains through time.

    Landlab component that simulates grading of soil particles through time
    based on mARM (Cohen et al., 2009, 2010) approach.

    The fragmentation process is controlled by a weathering transition matrix which
    defines the relative mass change in each soil grain size class (grading class)
    as a result of the fracturing of particles in the weathering mechanism.

    The primary method of this class is :func:`run_one_step`.

    References
    ----------
    Cohen, S., Willgoose, G., & Hancock, G. (2009). The mARM spatially distributed soil
    evolution model: A computationally efficient modeling framework and analysis of
    hillslope soil surface organization. Journal of Geophysical Research: Earth Surface,
    114(F3).

    Cohen, S., Willgoose, G., & Hancock, G. (2010). The mARM3D spatially distributed
    soil evolution model: Three‐dimensional model framework and analysis of hillslope
    and landform responses. Journal of Geophysical Research: Earth Surface, 115(F4).
    """

    _name = "SoilGrading"
    _unit_agnostic = True
    _info = {
        "topographic__elevation": {
            "dtype": float,
            "intent": "inout",
            "optional": True,
            "units": "m",
            "mapping": "node",
            "doc": "Topographic elevation at node",
        },
        "bedrock__elevation": {
            "dtype": float,
            "intent": "inout",
            "optional": True,
            "units": "m",
            "mapping": "node",
            "doc": "Bedrock elevation at node",
        },
        "soil__depth": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Soil depth at node",
        },
        "grains__mass": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "kg",
            "mapping": "node",
            "doc": "mass of grains in each size class stored at node",
        },
        "median_size__mass": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": (
                "The median grain size in each node based on soil grains"
                " mass distribution"
            ),
        },
        "grains_classes__size": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "The size of each grain size class",
        },
        "bed_grains__proportions": {
            "dtype": float,
            "intent": "out",
            "optional": True,
            "units": "m",
            "mapping": "node",
            "doc": "Proportional mass of each grain size class in bedrock",
        },
    }

    def __init__(
        self,
        grid,
        meansizes=(0.0001, 0.0005, 0.002),
        limits=None,
        grains_mass=None,
        fragmentation_pattern=(0, 1),
        A_factor=1,
        initial_median_size=None,
        soil_density=2650,
        phi=0.5,
        initial_total_soil_mass=2650,
        std=None,
        CV=0.6,
        is_bedrock_distribution_flag=False,
        seed=2024,
    ):
        """
        Parameters
        ----------
        grid : RasterModelGrid
            A grid.
        meansizes : float (m)
            The mean grain size in each size class.
        limits : float (m)
            2D array with the limits of each size class
        fragmentation_pattern : float (m)
            A list of floats describes how much mass transfer from a parent grain to
            daughters. The list must be in a size >=2 and < number of size classes,
            while the first element is percentage that remains in the parent grain.
            The other elements are the proportion of parent grain that is weatherd for
            each daughter size class. The sum of fragmentation pattern list should
            be <=1. Default value set to [0, 1] which means that the entire mass of
            parent grain is transffer to the next (smaller) size class.
        A_factor : float (-. 0-1), optional
            Factor that control the fragmentation rate.
        initial_median_size : float (m), optional
            The initial median grain size in the soil.
        grains_mass : float (m), optional
            The mass of each grain size class
        soil_density : float (kg/m^3), optional
            Density of the soil particles.
        phi : float (-. 0-1), optional
            Soil porosity.
        initial_total_soil_mass : float (kg), optional
            The total initial soil mass (taking into account all grain size classes).
        std: float, optional
            The standard deviation of grain size distribution.
        CV: float, optional
            Coefficient of variance of the grain size distribution.
        is_bedrock_distribution_flag: bool, optional
            A flag to indicate if the grain size distribution is generated for soil or
            bedrock layer.
        seed: float, optional
            Provide seed to set grain size distribution.
            If not provided, seed is set to 2024.
        """

        super().__init__(grid)
        self._grains_mass = grains_mass
        if ~np.all(np.isnan(self._grains_mass)):
            if isinstance(grains_mass, list):
                self._grains_mass = np.array(grains_mass)
            if isinstance(grains_mass, tuple):
                self._grains_mass = np.array(list(grains_mass))
            if np.ndim(self._grains_mass)==1:
                self._grains_mass = (np.ones((np.size(self._grid.nodes.flatten()),np.size(self._grains_mass))) *
                                       self._grains_mass[np.newaxis, :])

        self._meansizes = meansizes
        if isinstance(self._meansizes, list):
            self._meansizes = np.array(self._meansizes)
        elif isinstance(self._meansizes, tuple):
            self._meansizes = np.array(list(self._meansizes))

        self._limits = limits
        if isinstance(self._limits, list):
            self._limits = np.array(self._limits)
        if isinstance(self._limits, tuple):
            self._limits = np.array(list(self._limits))
        self._n_sizes = np.size(meansizes)
        self._fragmentation_pattern = fragmentation_pattern
        self._A_factor = A_factor
        self._soil_density = float(soil_density)
        self._phi = phi
        self._is_bedrock_distribution_flag = is_bedrock_distribution_flag
        self._seed = seed
        random.seed(seed)

        # Check if the fragmentation pattern provided is valid
        self.check_fragmentation_pattern()

        # Note: Landlabs' init_out_field procedure will not work
        # for the 'grains__mass' and 'grains_classes__size' fields
        # because the shape of these fields is: n_nodes x n_grain_sizes.
        grid.add_zeros("median_size__mass", at="node")

        # Create 2D arrays, ensuring they stay 2D even with single grain size
        grains_classes_array = np.ones((grid.number_of_nodes, self._n_sizes))
        grains_mass_array = np.zeros((grid.number_of_nodes, self._n_sizes))
        bed_proportions_array = np.ones((grid.number_of_nodes, self._n_sizes))

        # Prevent Landlab from squeezing single-column arrays to 1D
        grid.at_node["grains_classes__size"] = grains_classes_array.reshape(grid.number_of_nodes, -1)
        grid.at_node["grains__mass"] = grains_mass_array.reshape(grid.number_of_nodes, -1)
        grid.at_node["bed_grains__proportions"] = bed_proportions_array.reshape(grid.number_of_nodes, -1)

        # Create fields for soil depth, topographic elevation and bedrock elevation
        if grid.has_field("soil__depth"):
            warnings.warn(
                "Soil depth is rewrite due to inconsistent with grains__mass",
                stacklevel=2,
            )
        grid.add_zeros("soil__depth", at="node", clobber=True)
        if "topographic__elevation" not in grid.at_node:
            grid.add_zeros("topographic__elevation", at="node", clobber=True)

        if "bedrock__elevation" not in grid.at_node:
            grid.add_zeros("bedrock__elevation", at="node", clobber=True)

        # Update sizes and distribution limits
        self._grid.at_node["grains_classes__size"][self._grid.nodes] *= self._meansizes
        if np.size(self._limits) == 1 and self._limits is None:
            self.set_grading_limits()
        else:

            if (
                np.shape(self._limits)[0] != np.size(meansizes)
                or np.shape(self._limits)[1] != 2
            ):
                raise ValueError("limits array must be in shape of meansizes x 2")
            if (
                np.all(self._limits[:, 1] > self._limits[:, 0])
                * np.all(np.diff(self._limits[:, 1]) > 0)
                * np.all(np.diff(self._limits[:, 0]) > 0)
            ) is False:
                raise ValueError("limits array must in ascending order")

        # Transition matrix
        self.create_transition_mat()

        # Update the mass in each size class.
        # In case grains_mass not provided, the masss will be spread around
        # the initial_median_size asuuming normal distribution
        if self._grains_mass is None:
            self._CV = CV
            if initial_median_size is None:
                self._initial_median_size = self._meansizes[int(self._n_sizes / 2)]
            else:
                self._initial_median_size = initial_median_size
            if std is None:
                std = self._initial_median_size * self._CV
            self._std = std
            self._initial_total_soil_mass = initial_total_soil_mass
            self.generate_mass_distribution()
        else:
            if np.ndim(self._grains_mass) <= 1:
                if not np.size(self._grains_mass) == np.size(self._meansizes):
                    raise ValueError(
                        "grains_mass and meansizes do not have the same size"
                    )
            else:
                if not np.shape(self._grains_mass)[1] == np.size(meansizes):
                    raise ValueError(
                        "grains_mass and meansizes do not have the same size"
                    )


            self._update_mass(self._grains_mass, self._is_bedrock_distribution_flag)

        self.update_median_grain_size()

    @property
    def A(self):
        """Transition matrix."""
        return self._A

    @property
    def A_factor(self):
        """Fragmentation rate"""
        return self._A_factor

    @staticmethod
    def create_grading(
        grading_name,
        grain_max_size,
        power_of=1 / 3,
        n_sizes=10,
        precent_of_volume_in_spread=10,
    ):
        """
        Parameters
        ----------
        grading_name : str
             Grading name should be provided in the from pX-AAA-BBB-CCC-DDD where X is
             the total number of daughter particles the grading fractions have, AAAA
             is the percentage of volume that remains in the parent size fraction
             after fragmentation, BBB and CCC (and so on) are the proportion of
             daugther particles in the next (smaller) size classes.
         grain_max_size : float (m)
             The maximal grain size represented in the grading distribution
         power_of : float (-)
             The power relation between parent and dauther grain volume
         n_sizes : int (-)
             The number of classes represented in the distribution
         precent_of_volume_in_spread : float (-)
             The percentage of mass from parent spread between the daughter grains
              in case of 'spread' flag is found in the grading name.
        """

        # Make sure that grading_name is valid
        grading_name_split = grading_name.split("-")
        is_p = grading_name_split[0][0] == "p"
        is_size = np.size(grading_name_split) > 2
        is_N = len(grading_name_split[0]) == 2 and grading_name_split[0][1].isnumeric()
        is_numeric = True
        sum_values = 0
        for s in grading_name.split("-")[1:]:
            if s != "spread":
                is_numeric *= s.isnumeric()
                if is_numeric:
                    sum_values += int(s)
        is_smaller_than_100 = sum_values <= 100 or "spread" in grading_name
        if is_p * is_size * is_numeric * is_smaller_than_100 * is_N is False:
            raise ValueError("grading name provided not valid")

        # Create grading distribution based on the grading name
        N = int(grading_name_split[0][1])
        parent = np.copy(grain_max_size)
        limits = np.zeros((n_sizes, 2))
        limits[0, 1] = grain_max_size
        for cnt in range(0, n_sizes):
            limits[cnt, 1] = parent
            parent = parent * (1 / N) ** (power_of)
            limits[cnt, 0] = parent
        limits = limits[::-1]  # sort in ascend order
        meansizes = np.asarray((limits[:, 0] + limits[:, 1]) / 2)

        # Create fragmentation vector describing the fragmentation pattern
        precents = np.array(
            [
                float(s)
                for s in grading_name.split("-")
                if s.replace(".", "", 1).isdigit()
            ]
        )
        if "spread" in grading_name:
            mass_precent_to_spread = 100 - precents[0] - np.sum(precents[1:])
            precents_to_add = (
                np.ones((1, n_sizes - len(precents))) * precent_of_volume_in_spread
            )
            if np.sum(precents_to_add) > mass_precent_to_spread:
                precents_to_add *= mass_precent_to_spread / np.sum(precents_to_add)
            precents = np.append(precents, precents_to_add)
        fragmentation_pattern = precents / 100

        return meansizes, limits, fragmentation_pattern

    def check_fragmentation_pattern(self):
        # Special case: single grain size (no fragmentation needed)
        if len(self._meansizes) == 1:
            # For single grain size, fragmentation pattern doesn't apply
            # Set to [1] meaning all mass stays in the same class
            if len(self._fragmentation_pattern) != 1:
                self._fragmentation_pattern = np.array([1.0])
            return

        # Normal case: multiple grain sizes
        if (
            len(self._fragmentation_pattern) < 2
            or len(self._fragmentation_pattern) > len(self._meansizes)
            or np.sum(self._fragmentation_pattern) > 1.0
        ):
            raise ValueError("fragmentation pattern provided not valid")

    def create_transition_mat(self):
        """
        This procedure creates a transition matrix that control the
        weathering / fragmentation pattern. The matrix represent the
        proportion of mass that weathered from each size class to other classes
        """

        self._A = np.zeros((self._n_sizes, self._n_sizes))
        self._A_factor = np.ones_like(self._A) * self._A_factor

        # Special case: single grain size (no transitions)
        if self._n_sizes == 1:
            # Identity matrix - all mass stays in the same class
            self._A[0, 0] = 0.0  # No change (no weathering/fragmentation)
            return

        # Normal case: multiple grain sizes
        self._A[0, 0] = -round(
            1
            - self._fragmentation_pattern[0]
            - np.sum(self._fragmentation_pattern[1:]),
            10,
        )
        for i in range(1, self._n_sizes):

            self._A[i, i] = -(1 - (self._fragmentation_pattern[0]))
            cnti = i - 1  # rows
            cnt = 1
            while cnti >= 0 and cnt <= (len(self._fragmentation_pattern) - 1):
                if cnti == 0 and cnt <= (len(self._fragmentation_pattern) - 2):
                    self._A[cnti, i] = (1 - self._fragmentation_pattern[0]) - np.sum(
                        self._fragmentation_pattern[1:cnt]
                    )
                else:
                    self._A[cnti, i] = self._fragmentation_pattern[cnt]
                cnt += 1
                cnti -= 1

    def set_grading_limits(self):

        self._limits = (self._meansizes[:-1] + self._meansizes[1:]) * 0.5
        self._limits = np.insert(self._limits, 0, 0.0)
        self._limits = np.concatenate((self._limits, [np.inf]))

    def generate_mass_distribution(
        self, median_size=None, is_bedrock_distribution_flag=False
    ):
        """
        This procedure split the total soil mass between all size classes
        assuming normal distribution around the median size class. Different
        grain size distribution can be assigned to the initial soil layer and
        the bedrock layer
        """

        if not is_bedrock_distribution_flag:

            is_bedrock_distribution_flag = self._is_bedrock_distribution_flag
            median_size = self._initial_median_size
            total_soil_mass = self._initial_total_soil_mass
            grains_mass__distribution = self._generate_normal_distribution(
                median_size=median_size, total_soil_mass=total_soil_mass
            )

        else:
            total_bedrock_mass = 10000
            # SoilGrading assumes that bedrock thickness is unlimited
            # and only the bedrock elevation is tracked.
            # The variable total_bedrock_mass is set to a large enough number just for
            # generate grain size distribution without relation to bedrock thickness.
            if median_size is None:
                grains_mass__distribution = self.g_state0
            else:
                if self._std is None:
                    self._std = self._CV * median_size
                grains_mass__distribution = self._generate_normal_distribution(
                    median_size=median_size,
                    total_soil_mass=total_bedrock_mass,
                )
        self._update_mass(grains_mass__distribution, is_bedrock_distribution_flag)

    def _update_mass(self, grains_mass__distribution, is_bedrock_distribution_flag):
        self.g_state_bedrock = grains_mass__distribution

        # Get the bed proportions field
        bed_proportions = self._grid.at_node["bed_grains__proportions"]

        # Handle dimensionality - Landlab may squeeze arrays with single columns
        is_1d = bed_proportions.ndim == 1

        # Handle both 1D and 2D grain mass distributions
        if np.ndim(self.g_state_bedrock) > 1:
            # 2D case: (nodes, grain_sizes)
            if is_1d:
                # Single grain size - use 1D indexing
                bed_proportions[self._grid.core_nodes] = 1.0
            else:
                bed_proportions[self._grid.core_nodes, :] = 1
                bed_proportions[self._grid.core_nodes, :] *= np.divide(
                    self.g_state_bedrock[self._grid.core_nodes],
                    np.sum(self.g_state_bedrock, 1)[self._grid.core_nodes, np.newaxis]
                )
        else:
            # 1D case: single grain size or uniform distribution
            if is_1d:
                bed_proportions[self._grid.core_nodes] = 1.0
            else:
                bed_proportions[self._grid.core_nodes, :] = 1.0
                # For single grain size, proportion is 100% in that size
                if self._n_sizes == 1:
                    bed_proportions[self._grid.core_nodes, 0] = 1.0

        if not is_bedrock_distribution_flag:
            # self.g_state = np.full(
            #     self.grid.shape + (len(grains_mass__distribution),),
            #     grains_mass__distribution,
            # )

            self.g_state0 = grains_mass__distribution
            grains_mass = self._grid.at_node["grains__mass"]
            is_grains_1d = grains_mass.ndim == 1

            if is_grains_1d:
                # Single grain size - use 1D indexing
                # Flatten 2D distribution if needed (from (N,1) to (N,))
                if np.ndim(grains_mass__distribution) > 1:
                    grains_mass[self._grid.core_nodes] = grains_mass__distribution[self._grid.core_nodes, 0]
                else:
                    grains_mass[self._grid.core_nodes] = grains_mass__distribution[self._grid.core_nodes]
                layer_depth = grains_mass[self._grid.core_nodes] / (
                    self._soil_density * self._grid.dx * self._grid.dx
                )
            else:
                # Multiple grain sizes - use 2D indexing
                grains_mass[self._grid.core_nodes, :] = 1
                grains_mass[self._grid.core_nodes, :] = 1 * grains_mass__distribution[self._grid.core_nodes, :]
                layer_depth = np.sum(grains_mass[self._grid.core_nodes], 1) / (
                    self._soil_density * self._grid.dx * self._grid.dx
                )

            layer_depth /= 1 - self._phi

            self._grid.at_node["soil__depth"][self._grid.core_nodes] += layer_depth
            self._grid.at_node["topographic__elevation"] = (
                self._grid.at_node["soil__depth"]
                + self._grid.at_node["bedrock__elevation"]
            )

    def _generate_normal_distribution(self, median_size=None, total_soil_mass=None):

        if median_size is None:
            median_size = self._initial_median_size
        if self._std is None:
            self._std = self._CV * median_size
        if total_soil_mass is None:
            total_soil_mass = self._initial_total_soil_mass

        lower = np.min(self._limits)
        upper = np.max(self._limits)

        values = []
        if median_size < lower:
            grains_mass__distribution = np.zeros_like(self._meansizes)
            grains_mass__distribution[0] = total_soil_mass
            warnings.warn(
                "Median size requested is smaller than the smallest mean size"
                "in the distribution",
                stacklevel=2,
            )

        elif median_size > upper:
            grains_mass__distribution = np.zeros_like(self._meansizes)
            grains_mass__distribution[-1] = total_soil_mass
            warnings.warn(
                "Median size requested is larger than the largest mean size"
                "in the distribution",
                stacklevel=2,
            )

        else:
            while np.size(values) < total_soil_mass:
                sample = random.gauss(median_size, self._std)
                if sample >= lower and sample <= upper:
                    values.append(sample)

            grains_mass__distribution = np.histogram(values, self._limits)[0]

        return grains_mass__distribution

    def update_median_grain_size(self):
        """
        The median grain size at each node is defined as the size of the class closest
        to the median based on the mass in each size class
        """
        grains_mass = self._grid.at_node["grains__mass"]

        # Handle single grain size case
        if self._n_sizes == 1 or grains_mass.ndim == 1:
            # For single grain size, median is just that size
            self._grid.at_node["median_size__mass"][:] = self._meansizes[0]
            return

        # Multiple grain sizes case
        cumsum_gs = np.cumsum(grains_mass, axis=1)
        sum_gs = np.sum(grains_mass, axis=1)
        self._grid.at_node["median_size__mass"][sum_gs <= 0] = 0
        sum_gs_exp = np.expand_dims(sum_gs, -1)

        fraction_from_total = np.divide(
            cumsum_gs,
            sum_gs_exp,
            out=np.zeros_like(cumsum_gs),
            where=sum_gs_exp != 0,
        )
        fraction_from_total[fraction_from_total < 0.5] = np.inf
        median_val_indx = np.argmin(
            fraction_from_total - 0.5,
            axis=1,
        )

        self._grid.at_node["median_size__mass"][self._grid.core_nodes] = (
            self._meansizes[median_val_indx[self._grid.core_nodes]]
        )

    def run_one_step(self, A_factor=None):

        if np.any(A_factor is None):
            A_factor = self._A_factor

        # Special case: single grain size class. The transition matrix `A`
        # is forced to all zeros in create_transition_mat() when there is
        # only one size class (no fragmentation is possible between
        # classes), so the block below is always a no-op mathematically.
        # Skipping it also avoids a shape bug: when n_sizes == 1, Landlab
        # squeezes the "grains__mass" field down to a 1D (n_nodes,) array
        # instead of keeping it 2D (n_nodes, 1), which made the
        # `self._grid.at_node["grains__mass"] += np.reshape(...)` line
        # below broadcast to an (n_nodes, n_nodes) shape and raise a
        # ValueError.
        if self._n_sizes == 1:
            self.update_median_grain_size()
            return

        temp_g_mass = np.moveaxis(
            np.dot(
                self._A * A_factor,
                np.swapaxes(
                    np.reshape(
                        self._grid.at_node["grains__mass"][self._grid.nodes],
                        (self._grid.shape[0], self._grid.shape[1], self._n_sizes),
                    ),
                    1,
                    2,
                ),
            ),
            0,
            -1,
        )

        self._grid.at_node["grains__mass"] += np.reshape(
            temp_g_mass, (self._grid.shape[0] * self._grid.shape[1], self._n_sizes)
        )
        self.update_median_grain_size()
