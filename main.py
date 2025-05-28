import matplotlib.pyplot as plt
import numpy as np
from funcs.slab_failures import SlabFailures
from funcs.OverlandflowErosionDeposition import OverlandflowErosionDeposition
from funcs.SoilInfiltrationGreenAmpt_YS import SoilInfiltrationGreenAmpt
from funcs.soil_grading import SoilGrading
from landlab.components import OverlandFlow
from landlab.components import PriorityFloodFlowRouter
from landlab.io import read_esri_ascii
from funcs.GradMapper import GradMapper
from landlab import imshow_grid

## Model parameters
roughness = 0.07
Ks = 5.5*10**-6
kr = 0.0005
soil_type = 'sandy loam'
phi = 0.4
soil_density = 2650

## Load DEM
grid_path = './Inputs/LuckyHills103_1m.asc'
outlet_node = int(14504)

grid, data = read_esri_ascii(grid_path)
grid.set_watershed_boundary_condition(node_data=data, nodata_value=-9999.0)
#outlet_node = int(np.where(grid._node_status==1)[0])

## Update elevation field
grid.add_zeros('topographic__elevation', at='node')
grid.add_zeros('bedrock__elevation', at='node')

topo = grid.at_node['topographic__elevation']
topo[:] = data
bedrock = grid.at_node['bedrock__elevation']
bedrock[:] = np.copy(topo)

## Load components
## SoilGrading (track multiple grain size classes)
meansizes = [0.001, 0.01, 0.1]
sg = SoilGrading(grid,
            meansizes=meansizes,
            grains_weight=[1000, 1000, 1000],
                 phi = phi, soil_density = soil_density)


## Overlandflow
grid.add_zeros('water_surface__slope',
                       at="node")
grid.add_zeros('surface_water__depth',
               at="node")
grid.add_zeros('surface_water__depth_at_link',
               at="link")
grid.add_zeros('water_surface__elevation',
               at="node")
grid.at_link['surface_water__depth_at_link'][:]  = 10**-8
grid.at_node['surface_water__depth'][:]  = 10**-8
of = OverlandFlow(grid, mannings_n=roughness,
                  steep_slopes=True,
                  alpha = 0.7)


## Infiltration
infilitration_depth = grid.add_ones("soil_water_infiltration__depth", at="node", dtype=float)
infilitration_depth *= 0.001  ## meter
SI = SoilInfiltrationGreenAmpt(grid, hydraulic_conductivity=Ks,
                                     soil_type=soil_type)


## Overlandflow erosion/deposition
dspe = OverlandflowErosionDeposition(
            grid,
    slope='water_surface__slope',
    kr= kr,
    change_topo_flag=True,
    phi=phi,
    sigma = soil_density)


## PriorityRouter
fr = PriorityFloodFlowRouter(
    grid,
    flow_metric="D8",
    separate_hill_flow=True,
    hill_flow_metric="Quinn",
    update_hill_flow_instantaneous=True, depression_handler='fill'
)
fr.run_one_step()


## Failures
slab_failures = SlabFailures(grid)

## Mapper
gradmap = GradMapper(grid=grid)


# Read rainfall data
rainfall_data = np.load('./Inputs/rainfall_data.npz')
rainfall_duration = rainfall_data['durations']  # Seconds
rainfall_rate = rainfall_data['rates']          # Rainfall intensity [m/s]


## Simulation parameters
epsilon = 10**-10       # Small value because water depth cannot be zero
elapse_dts = 0          # Counter of simulation time [sec]
saving_resolution = 30  # Sec
min_dt = 30             # Maximal dt [sec] to ensure stability

## List for saving
Q_at_node = []
elapsed_dt_vec = []
sediment_weight_flux_kg_s = []
rainfall_vec = []
ms_to_mmh = 60*60*1000  # Convert m/s to mm/h

# Update pointers
topo = grid.at_node['topographic__elevation']
bedrock = grid.at_node['bedrock__elevation']
# Lets save the topography before the storm
topo_init = np.copy(topo)

# Main loop
n_repeats = 1
for _ in range(n_repeats):
    fr.run_one_step()
    slab_failures.run_one_step()
    while elapse_dts < rainfall_duration[-1]:
        if elapse_dts ==0:
            int_index = 0
            of.rainfall_intensity = rainfall_rate[int_index]
            current_rainfall_rate = rainfall_rate[0]
            current_rainfall_duration = rainfall_duration[0]
            cnt_saving = 0

        if elapse_dts >= current_rainfall_duration:  # sec

            int_index += 1
            current_rainfall_rate = rainfall_rate[int_index]
            current_rainfall_duration = rainfall_duration[int_index]
            of.rainfall_intensity = current_rainfall_rate  # meter per sec -> For the OverlandFlow component


        gradmap.run_one_step()
        of.calc_time_step()
        dspe.calc_rates()


        dt = np.min((of.calc_time_step(),
                     min_dt,
                     dspe._stable_dt,
                     current_rainfall_duration-elapse_dts))

        SI.run_one_step(dt=dt)
        of.run_one_step(dt=dt)
        dspe.run_one_step_basic(dt=dt)

        if elapse_dts > cnt_saving:
            discharge_at_node = of.discharge_mapper(grid.at_link['surface_water__discharge'])
            total_sediment_incoming_flux = gradmap.flux_mapper()

            Q_at_node.append(discharge_at_node[outlet_node ])
            sediment_weight_flux_kg_s.append(total_sediment_incoming_flux[outlet_node ])  # kg/s
            elapsed_dt_vec.append(elapse_dts)
            rainfall_vec.append(current_rainfall_rate)
            cnt_saving += 30

        elapse_dts += dt
        print(elapse_dts)

    # Let the watershed run-out of water
    of.rainfall_intensity = epsilon
    while elapse_dts < rainfall_duration[-1] * 1.8:

        gradmap.run_one_step()
        of.calc_time_step()
        dspe.calc_rates()

        dt = np.min((of.calc_time_step(),
                     min_dt,
                     dspe._stable_dt))

        SI.run_one_step(dt=dt)
        of.run_one_step(dt=dt)
        dspe.run_one_step_basic(dt=dt)

        if elapse_dts > cnt_saving:
            discharge_at_node = of.discharge_mapper(grid.at_link['surface_water__discharge'])
            total_sediment_incoming_flux = gradmap.flux_mapper()

            Q_at_node.append(discharge_at_node[outlet_node ])
            sediment_weight_flux_kg_s.append(total_sediment_incoming_flux[outlet_node ])  # kg/s
            elapsed_dt_vec.append(elapse_dts)
            rainfall_vec.append(current_rainfall_rate)

            cnt_saving+=30


        elapse_dts += dt
        print(elapse_dts)

## Plot the results
# First, lets plot differencing map
imshow_grid(grid, topo_init-topo,
            vmax=0.05, vmin=0.)
plt.show()

# Now, lets plot hydrograph/sedigraph at the outlet
fig, ax = plt.subplots(figsize=(14,11))
rainfall_vec_mmh = np.array(rainfall_vec)*ms_to_mmh
elapsed_dt_vec_minutes = np.array(elapsed_dt_vec)/60
Q_rate_at_node = np.array(Q_at_node)*60*60 # sec to hour
Q_rate_at_node /= np.size(topo[topo>0])*grid.dx*grid.dx # m^3 to m
Q_rate_at_node *=1000 # m to mm
ax.plot(elapsed_dt_vec_minutes, Q_rate_at_node, color='blue', linewidth=6)
ax2  = ax.twinx()
ax2.plot(elapsed_dt_vec_minutes, sediment_weight_flux_kg_s,color='green', linewidth=6)
ax.set_ylim([0,20])

ax3 = ax.twinx()
ax3.bar(elapsed_dt_vec_minutes,
        rainfall_vec_mmh,
        width=3,
        color='black')

ax2.set_yticks([0, 0.5, 1, 1.5])
ax2.set_ylim([0,4])
ax3.set_ylim([0,350])
ax3.set_yticks([0, 50, 100, 150])
ax3.invert_yaxis()
fs=45
ax.set_xlabel('Time [min]', fontsize=fs)
ax.set_ylabel('Runoff rate [mm/h]', fontsize=fs)
ax2.set_ylabel('Sediment \n flux [kg/s]', fontsize=fs)
ax2.yaxis.set_label_coords(1.15,0.2)

ax3.set_ylabel('Rainfall intensity\n  [mm/h]', fontsize=fs)
ax3.yaxis.set_label_coords(1.15,0.75)

ax.tick_params(axis='both', labelsize=fs)
ax2.tick_params(axis='both', labelsize=fs)
ax3.tick_params(axis='both', labelsize=fs)
plt.tight_layout()
plt.savefig('results.jpg', dpi=300)
plt.show()


