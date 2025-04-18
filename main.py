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

## Model parameters
roughness = 0.07
Ks = 5.5*10**-6
kr = 0.0005
soil_type = 'sandy loam'


## Load DEM
grid_path = './Inputs/LuckyHills103_1m.asc'
outlet_node = int(14504)

#grid_path = '../Inputs/LuckyHills103_10m.asc'

grid, data = read_esri_ascii(grid_path)
grid.set_watershed_boundary_condition(node_data=data, nodata_value=-9999.0)
#outlet_node = int(np.where(grid._node_status==1)[0])

## Update elevation field
topo = grid.add_zeros('topographic__elevation', at='node')
bedrock = grid.add_zeros('bedrock__elevation', at='node')
topo[:] = data
bedrock[:] = topo[:]

## Load components
## SoilGrading (track multiple grain size classes)
meansizes = [0.001, 0.01, 0.1]
sg = SoilGrading(grid,
            meansizes=meansizes,
            grains_weight=[1000, 1000, 1000])


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
    change_topo_flag=True) 


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

## Main loop
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



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import hydroeval as he
import pickle
from scipy.interpolate import griddata
import os
from scipy import stats

observed_table_path = '/Users/yush9908/Dropbox/Mac/Documents/GullyProject/LH103_data/Lucky_Hills_103_sediment_events.csv'
calib_path = '/Users/yush9908/Dropbox/Mac/Documents/GullyProject/Outputs/Calibration/'
calib_out_folder = '/Users/yush9908/Dropbox/Mac/Documents/GullyProject/Outputs/Calibration/calib_dfs'
figures_folder  = '/Users/yush9908/Dropbox/Mac/Documents/GullyProject/Outputs/Figures'


## Load currect calib simulation
observed_table = pd.read_csv(observed_table_path )
df_calib = pd.DataFrame(columns=['set n', 'KE', 'mannings n','nse total runoff','nse peak runoff','nse sediment','set_data'])
calib_data = {}
# iterate over files in
# that directory
directory = '/Users/yush9908/Dropbox/Mac/Documents/GullyProject/Outputs/Calibration/final'
for n,filename in enumerate(os.listdir(directory)):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if not filename.startswith('.') and os.path.isfile(f):
        print(f)
        set_data_dic = {}

        calib_table = pd.read_csv(f)

        observed_tot_runoff = observed_table['obsRunoff']
        calc_tot_runoff = calib_table['total_out_runoff']

        observed_peak_runoff = observed_table['obsPeak']
        calc_peak_runoff = calib_table['peak_discharge_outlet_mm_h']

        nse = he.evaluator(he.nse, calc_tot_runoff.values, observed_tot_runoff[calib_table.event_index_in_table.values].values)
        kge, r, alpha, beta = he.evaluator(he.kge, calc_tot_runoff.values, observed_tot_runoff[calib_table.event_index_in_table.values].values)

        nse_peak = he.evaluator(he.nse, calc_peak_runoff.values, observed_peak_runoff[calib_table.event_index_in_table.values].values)

        calib_table['observed_runoff'] = observed_tot_runoff[calib_table.event_index_in_table.values].values
        calib_table['observed_peak_runoff'] = observed_peak_runoff[calib_table.event_index_in_table.values].values
        calib_table['observed_sediment'] = observed_table['obsSediKg'][calib_table.event_index_in_table.values].values


        calib_data[f'set_{n}'] =  calib_table
        df_calib.loc[n,'set n'] = n
        df_calib.loc[n, 'KE'] =  calib_table.loc[0,'KE']
        df_calib.loc[n, 'mannings n'] = calib_table.loc[0,'n']
        df_calib.loc[n,'nse total runoff'] = nse
        df_calib.loc[n,'nse peak runoff'] = nse_peak

# this is for the current calib sets
calib_data['sum_df'] =  df_calib

df_calib.sort_values(by=['KE','mannings n'], inplace=True)
from scipy.interpolate import interp2d

x = np.array(df_calib['mannings n'].values,dtype='float64')
y = np.array(df_calib['KE'].values,dtype='float64')
#z = np.array(df_calib['nse total runoff'].to_list(),dtype='float64')

z= (np.array(df_calib['nse total runoff'].to_list(),dtype='float64') + np.array(df_calib['nse peak runoff'].to_list(),dtype='float64'))/2
X,Y= np.meshgrid(x,y)
Z = griddata((x, y), z, (X, Y),method='linear')
cmap = plt.colormaps["winter"].with_extremes(under="magenta", over="yellow")


calib_tabel_path = '/Users/yush9908/Dropbox/Mac/Documents/GullyProject/Outputs/Calibration/calib_data_test_92.csv'
figures_folder  = '/Users/yush9908/Dropbox/Mac/Documents/GullyProject/Outputs/Figures'
calib_table = pd.read_csv(calib_tabel_path)
observed_table = pd.read_csv(observed_table_path )

observed_tot_runoff = observed_table['obsRunoff'][0:np.shape(calib_table)[0]]
calc_tot_runoff = calib_table['total_out_runoff']

observed_peak_runoff = observed_table['obsPeak'][0:np.shape(calib_table)[0]]
calc_peak_runoff = calib_table['peak_discharge_outlet_mm_h']
calc_sedi = calib_table['suspended_sediment_out_kg']
observed_sedi = observed_table['obsSediKg'][0:np.shape(calib_table)[0]]


observed_tot_runoff=observed_tot_runoff[~np.isnan(calc_tot_runoff)]
calc_tot_runoff=calc_tot_runoff[~np.isnan(calc_tot_runoff)]

calib_events = [17,50,47,13,37,61,85,57,91,72,23,51,77,24,20,27,4,92]
for i  in np.arange(0,12).tolist(): # this is to remove the old data
    calib_events.append(i)

all_evens = np.arange(0,93)
all_evens = np.delete(all_evens,calib_events)
nse = he.evaluator(he.nse, calc_tot_runoff.values[all_evens], observed_tot_runoff.values[all_evens])
kge, r, alpha, beta = he.evaluator(he.kge,  calc_tot_runoff.values[all_evens], observed_tot_runoff.values[all_evens])

res = stats.spearmanr(calc_tot_runoff.values[all_evens], observed_tot_runoff.values[all_evens])
spearman_runoff = res.statistic

nse_peak = he.evaluator(he.nse, calc_peak_runoff.values[all_evens], observed_peak_runoff.values[all_evens])
kge, r_peak, alpha, beta = he.evaluator(he.kge,  calc_tot_runoff.values[all_evens], observed_tot_runoff.values[all_evens])
res = stats.spearmanr(calc_peak_runoff.values[all_evens], observed_peak_runoff.values[all_evens])
spearman_peak_runoff = res.statistic


nse_sedi = he.evaluator(he.nse, calc_sedi.values[all_evens], observed_sedi.values[all_evens])
kge, r_sedi, alpha, beta = he.evaluator(he.kge,  calc_sedi.values[all_evens], observed_sedi.values[all_evens])
res = stats.spearmanr(calc_sedi.values[all_evens], observed_sedi.values[all_evens])
spearman_sediment = res.statistic


marker_size =  35
alpha = 0.5
markersize=35
fontsize=32

fig,ax = plt.subplots(2,3,
                      figsize=(30,15))
ax[0,0].plot(observed_tot_runoff[11:],calc_tot_runoff[11:],'.',label='',
           color='black', markersize=markersize),
ax[0,0].plot([0,40],[0,40],'-',label='',color='black')
ax[0,0].set_xlabel('Observed runoff [mm]',fontsize=fontsize)
ax[0,0].set_ylabel('Simulated runoff [mm]',fontsize=fontsize)
ax[0,0].tick_params(axis='both', which='major', labelsize=fontsize)
ax[0,0].text(0.65,0.1, R'$r^{2}$ ' +' = ' + str(np.round(r[0]**2,2)) +
        '\nNSE = ' + str(np.round(nse[0],2)) +
             '\n' + "\u03C1" + ' = ' + str(np.round(spearman_runoff,2)),
             transform=ax[0,0].transAxes,
             fontsize=25,bbox = {'facecolor':'white'})
ax[0,0].set_xlim([0,45])
ax[0,0].set_ylim([0,45])



ax[0,1].plot(observed_peak_runoff[11:],calc_peak_runoff[11:],'.',label='',
           color='black', markersize=markersize),
ax[0,1].plot([0,100],[0,100],'-',label='',color='black')
ax[0,1].set_xlabel('Observed peak runoff [mm/h]',fontsize=fontsize)
ax[0,1].set_ylabel('Simulated peak runoff [mm/h]',fontsize=fontsize)
ax[0,1].set_xlim([0,100])
ax[0,1].set_ylim([0,100])
ax[0,1].plot([0,100],[0,100],'-',label='',color='black')

ax[0,1].tick_params(axis='both', which='major', labelsize=20)
ax[0,1].text(0.65,0.1, R'$r^{2}$ ' +' = ' + str(np.round(r_peak[0]**2,2)) +
        '\nNSE = ' + str(np.round(nse_peak[0],2)) +
             '\n' + "\u03C1" + ' = ' + str(np.round(spearman_peak_runoff,2)),
             transform=ax[0,1].transAxes, fontsize=25,bbox = {'facecolor':'white'})
#spearman_runoff, spearman_peak_runoff, spearman_sediment


ax[0,2].plot(observed_sedi[11:],calc_sedi[11:],'.',label='',
           color='black', markersize=markersize),
ax[0,2].plot([0,40],[0,40],'-',label='',color='black')
ax[0,2].set_xlabel('Observed sediment [kg]',fontsize=fontsize)
ax[0,2].set_ylabel('Simulated sediment [kg]',fontsize=fontsize)

ax[0,2].text(0.65,0.1, R'$r^{2}$ ' +' = ' + str(np.round(r_sedi[0]**2,2)) +
        '\nNSE = ' + str(np.round(nse_sedi[0],2)) +
             '\n' + "\u03C1" + ' = ' + str(np.round(spearman_sediment,2))
             , transform=ax[0,2].transAxes,
             fontsize=25,bbox = {'facecolor':'white'})
ax[0,2].set_xlim([0,30000])
ax[0,2].set_ylim([0,30000])
ax[0,2].plot([0,40000],[0,40000],'-',label='',color='black')
for n in range(3):
    ax[0,n].tick_params(axis='both', which='major', labelsize=fontsize,
                      width=4,
                      length=4)


ax[1,0].scatter(observed_tot_runoff,observed_peak_runoff,s=400,color='black',alpha = alpha,
           edgecolors='black',linewidth=3,label = 'Observed'),
ax[1,0].scatter(calc_tot_runoff,calc_peak_runoff,s=400,color='red',alpha = alpha,
           edgecolors='black',linewidth=3,label = 'Simulated'),
ax[1,0].set_xlabel('Runoff [mm]',fontsize=fontsize)
ax[1,0].set_ylabel('Peak runoff [mm/h]',fontsize=fontsize)
ax[1,0].tick_params(axis='both', which='major', labelsize=fontsize)
ax[1,0].legend(loc='lower right',fontsize=25)

ax[1,1].scatter(observed_tot_runoff,observed_sedi,s=400,color='black',alpha = alpha,
           edgecolors='black',linewidth=3,label = 'Observed',
                ),

ax[1,1].scatter(calc_tot_runoff,calc_sedi,s=400,color='red',alpha = alpha,
           edgecolors='black',linewidth=3,label = 'Simulated',
                ),
ax[1,1].set_xlabel('Runoff [mm]',fontsize=fontsize)
ax[1,1].set_ylabel('Sediment [kg]',fontsize=fontsize)
ax[1,1].tick_params(axis='both', which='major', labelsize=fontsize)


ax[1,2].scatter(observed_peak_runoff,observed_sedi,s=400,color='black',alpha = alpha,
           edgecolors='black',linewidth=3),
# ax[2].plot(observed_peak_runoff,observed_sedi,'o',label='',color='black',
#            alpha = alpha, markersize=marker_size),
ax[1,2].set_xlabel('Peak runoff [mm/h]',fontsize=fontsize)
ax[1,2].set_ylabel('Sediment [kg]',fontsize=fontsize)
ax[1,2].scatter(calc_peak_runoff,calc_sedi,s=400,color='red',alpha = alpha,
           edgecolors='black',linewidth=3),
ax[1,2].tick_params(axis='both', which='major', labelsize=fontsize)
for n in range(3):
    ax[1,n].tick_params(axis='both', which='major', labelsize=fontsize,
                      width=4,
                      length=4)

for axis in ['top','bottom','left','right']:
    ax[1,0].spines[axis].set_linewidth(2)
    ax[1,1].spines[axis].set_linewidth(2)
    ax[1, 2].spines[axis].set_linewidth(2)
    ax[0,0].spines[axis].set_linewidth(2)
    ax[0,1].spines[axis].set_linewidth(2)
    ax[0, 2].spines[axis].set_linewidth(2)
plt.tight_layout()
plt.savefig(figures_folder+'/CalibrationRegressions.png',dpi=500)
plt.show()

