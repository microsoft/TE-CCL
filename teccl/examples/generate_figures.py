from os import read
import seaborn as sns
import pandas as pd
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from matplotlib import pyplot as plt
from pathlib import Path


OUTPUT_DIR = Path("./experiments_output/figures/")
INPUT_DIR = Path("./experiments_output/tables/individual_tables/")

def faster_pct(teccl_lst, taccl_lst):
    faster_cnt = 0
    for i, teccl in enumerate(teccl_lst):
        if teccl > taccl_lst[i]:
            faster_cnt += 1
    print(faster_cnt, len(teccl_lst))
    return faster_cnt * 100 / len(teccl_lst)

def read_alg_bnw(file_name: str):
    df = pd.read_csv(INPUT_DIR / Path(file_name))
    return df['Algorithm Bandwidth (GB/s)']

def read_collective_time(file_name: str):
    df = pd.read_csv(INPUT_DIR / Path(file_name))
    return df['Collective Finish Time (us)']

def plot_taccl_comparison():

    ###########################################################################NDv2 solution quality###################################################################################
    transfer_size = [1 * 1000, 0.256 * 1000, 0.064 * 1000, 0.016 * 1000, 0.004 * 1000, 0.001 * 1000, 0.000256 * 1000, 6.40E-05 * 1000, 1.60E-05 * 1000, 4.00E-06 * 1000, 1.00E-06 * 1000]
    transfer_size = ['1G', '256M', '64M', '16M', '4M', '1M', '256K', '64K', '16K', '4K', '1K']
    TECCL_allgather_NDv2_2Chassis_alg_bnw = read_alg_bnw("NDv2_2_chassis_AllGather_Fast.csv")
    TECCL_allgather_NDv2_2Chassis_alg_bnw_early_stop = read_alg_bnw("NDv2_2_chassis_AllGather_Fast_Early_Stop.csv")
    TACCL_allgather_NDv2_2Chassis_alg_bnw  = [18.59887254, 20.48885118, 20.42640112, 0, 18.47575058, 16.09010459, 10.13460016, 4.8929664, 1.2618297, 0.3375527, 0.0984373]
    TECCL_allgather_NDv2_4Chassis_alg_bnw = read_alg_bnw("NDv2_4_chassis_AllGather_Fast.csv")
    TECCL_allgather_NDv2_4Chassis_alg_bnw_early_stop = read_alg_bnw("NDv2_4_chassis_AllGather_Fast_Early_Stop.csv")
    TACCL_allgather_NDv2_4Chassis_alg_bnw = [0, 15.62948356, 15.18458764, 14.96305995, 12.7064803, 8.302200083, 3.505888798, 1.05890139, 0.304645849, 0.077287219, 0.067381897]
    

    speedup_allgather_NDv2_2Chassis_normal = []
    speedup_allgather_NDv2_2Chassis_early_stop = []
    speedup_allgather_NDv2_4Chassis_normal = []
    speedup_allgather_NDv2_4Chassis_early_stop = []
    early_stop_2 =[]
    normal_2 = []
    early_stop_4 = []
    normal_4 = []
    infeasible_instances = []
    for i in range(len(transfer_size)):
        if TACCL_allgather_NDv2_2Chassis_alg_bnw[i] == 0 and len(speedup_allgather_NDv2_2Chassis_normal) > 0:
            infeasible_instances += [(transfer_size[i], speedup_allgather_NDv2_2Chassis_normal[-1])]
            speedup_allgather_NDv2_2Chassis_normal += [speedup_allgather_NDv2_2Chassis_normal[-1]]
            infeasible_instances += [(transfer_size[i], speedup_allgather_NDv2_2Chassis_early_stop[-1])]
            speedup_allgather_NDv2_2Chassis_early_stop += [speedup_allgather_NDv2_2Chassis_early_stop[-1]]
        elif TACCL_allgather_NDv2_2Chassis_alg_bnw[i] == 0:
            infeasible_instances += [(transfer_size[i], 100)]
            speedup_allgather_NDv2_2Chassis_normal += [100]
            speedup_allgather_NDv2_2Chassis_early_stop += [100]
        else:
            speedup_allgather_NDv2_2Chassis_normal += [(TECCL_allgather_NDv2_2Chassis_alg_bnw[i] -TACCL_allgather_NDv2_2Chassis_alg_bnw[i]) * 100/TACCL_allgather_NDv2_2Chassis_alg_bnw[i]]
            speedup_allgather_NDv2_2Chassis_early_stop += [(TECCL_allgather_NDv2_2Chassis_alg_bnw_early_stop[i] - TACCL_allgather_NDv2_2Chassis_alg_bnw[i]) * 100/TACCL_allgather_NDv2_2Chassis_alg_bnw[i]]
        early_stop_2 += ['2 ch, AG, ES']
        normal_2 += ['2 ch, AG, opt']
        early_stop_4 += ['4 ch, AG, ES']
        normal_4 += ['4 ch, AG, opt']
        if TACCL_allgather_NDv2_4Chassis_alg_bnw[i] == 0 and len(speedup_allgather_NDv2_4Chassis_early_stop) > 0:
            infeasible_instances += [(transfer_size[i], speedup_allgather_NDv2_4Chassis_normal[-1])]
            speedup_allgather_NDv2_4Chassis_normal += [speedup_allgather_NDv2_4Chassis_normal[-1]]
            infeasible_instances += [(transfer_size[i], speedup_allgather_NDv2_4Chassis_early_stop[-1])]
            speedup_allgather_NDv2_4Chassis_early_stop += [speedup_allgather_NDv2_4Chassis_early_stop[-1]]
            
        elif TACCL_allgather_NDv2_4Chassis_alg_bnw[i] == 0:
            infeasible_instances += [(transfer_size[i], 100)]
            speedup_allgather_NDv2_4Chassis_normal += [100]
            speedup_allgather_NDv2_4Chassis_early_stop += [100]
        else:
            speedup_allgather_NDv2_4Chassis_normal += [(TECCL_allgather_NDv2_4Chassis_alg_bnw[i]- TACCL_allgather_NDv2_4Chassis_alg_bnw[i]) * 100/ TACCL_allgather_NDv2_4Chassis_alg_bnw[i]]
            speedup_allgather_NDv2_4Chassis_early_stop += [(TECCL_allgather_NDv2_4Chassis_alg_bnw_early_stop[i] - TACCL_allgather_NDv2_4Chassis_alg_bnw[i]) * 100/TACCL_allgather_NDv2_4Chassis_alg_bnw[i]]


    NDv2_2_chassis_allgather_early_stop = pd.DataFrame(list(zip(transfer_size, speedup_allgather_NDv2_2Chassis_early_stop, early_stop_2 )), columns = ['transfer size', 'perf improvement', 'experiment']) 
    NDv2_2_chassis_allgather = pd.DataFrame(list(zip(transfer_size, speedup_allgather_NDv2_2Chassis_normal, normal_2)), columns = ['transfer size', 'perf improvement', 'experiment'])
    NDv2_4_chassis_allgather = pd.DataFrame(list(zip(transfer_size, speedup_allgather_NDv2_4Chassis_normal, normal_4 )), columns = ['transfer size', 'perf improvement', 'experiment'])
    NDv2_4_chassis_allgather_early_stop = pd.DataFrame(list(zip(transfer_size, speedup_allgather_NDv2_4Chassis_early_stop, early_stop_4)), columns = ['transfer size', 'perf improvement', 'experiment'])

#***************alltoall*****************

    TECCL_NDv2_2_chassis_alltoall_bw = read_alg_bnw("NDv2_2_chassis_AlltoAll_Fast.csv")
    TACCL_NDv2_2_chassis_alltoall_bw = [3.124517653, 3.123314813, 3.117085525, 3.098133375, 3.020691738, 2.78551532, 2.212236432, 1.271354787, 0.447427293, 0.12455239, 0.027174836]
    TECCL_NDv2_4_chassis_alltoall_bw = [2.06918849, 2.073210236, 2.073210236, 2.073210236, 2.053256336, 2.037567654, 1.993092124, 1.828049129, 1.307189542, 0.448933782, 0.180913614]
    TACCL_NDv2_4_chassis_alltoall_bw = [2.081371641, 2.075953966, 2.054297655, 1.968697706, 1.701258932, 0, 0.767662229, 0.280308339, 0.076016724, 0.022813146, 0.00599224]

    speedup_NDv2_2Chassis_alltoall = []
    speedup_NDv2_4Chassis_alltoall = []
    normal_2_alltoall = []
    normal_4_alltoall = []
    for i in range(len(TACCL_NDv2_2_chassis_alltoall_bw)):
        normal_2_alltoall += ['2 ch, AtoA']
        speedup_NDv2_2Chassis_alltoall += [(TECCL_NDv2_2_chassis_alltoall_bw[i] - TACCL_NDv2_2_chassis_alltoall_bw[i]) * 100 / TACCL_NDv2_2_chassis_alltoall_bw[i]]
        normal_4_alltoall += ['4 ch, AtoA']
        if TACCL_NDv2_4_chassis_alltoall_bw[i] > 0:
            speedup_NDv2_4Chassis_alltoall += [(TECCL_NDv2_4_chassis_alltoall_bw[i] - TACCL_NDv2_4_chassis_alltoall_bw[i]) * 100/TACCL_NDv2_4_chassis_alltoall_bw[i]]
        elif len(speedup_NDv2_4Chassis_alltoall) > 0:
            infeasible_instances += [(transfer_size[i], speedup_NDv2_4Chassis_alltoall[-1])]
            speedup_NDv2_4Chassis_alltoall += [speedup_NDv2_4Chassis_alltoall[-1]]
        else:
            infeasible_instances += [(transfer_size[i], 100)]
            speedup_NDv2_4Chassis_alltoall += [100]

    speedup_NDv2_2Chassis_alltoall = pd.DataFrame(list(zip(transfer_size, speedup_NDv2_2Chassis_alltoall, normal_2_alltoall)), columns = ['transfer size', 'perf improvement', 'experiment'])
    speedup_NDv2_4Chassis_alltoall = pd.DataFrame(list(zip(transfer_size, speedup_NDv2_4Chassis_alltoall, normal_4_alltoall)), columns = ['transfer size', 'perf improvement', 'experiment'])

    speedup_NDv2_alltoall_bw = pd.concat([speedup_NDv2_2Chassis_alltoall,speedup_NDv2_4Chassis_alltoall ])
    speedup_NDv2_alltoall_bw = speedup_NDv2_alltoall_bw.reset_index()

    speedup_ndv2 = pd.concat([NDv2_2_chassis_allgather_early_stop, NDv2_2_chassis_allgather, NDv2_4_chassis_allgather_early_stop, NDv2_4_chassis_allgather])
    speedup_ndv2 = speedup_ndv2.reset_index()
    
    print(speedup_ndv2.groupby(['experiment']).max())


    allgather_NDv2_2Chassis = pd.concat([NDv2_2_chassis_allgather_early_stop, NDv2_2_chassis_allgather, NDv2_4_chassis_allgather_early_stop, NDv2_4_chassis_allgather, speedup_NDv2_2Chassis_alltoall, speedup_NDv2_4Chassis_alltoall])
    allgather_NDv2_2Chassis = allgather_NDv2_2Chassis.reset_index()
    plt.figure().clear()
    myplot = sns.relplot(x='transfer size', y='perf improvement', 
                data=allgather_NDv2_2Chassis, hue = 'experiment',  style= 'experiment', kind="line", ci=95, palette="muted",linewidth = 7, height=8, aspect=11.7/8.27)
  
    legend = plt.legend( loc="upper left", frameon=True, fontsize=24)
    for line in legend.get_lines():
        line.set_linewidth(6)
    plt.xlabel("Output buffer size", fontsize=40)
    plt.ylabel("Improvement in algo bandwidth (%)", fontsize=29)
    sns.despine(left=False, right =False, top=False, bottom=False)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    sns.despine(left=False, right =False, top=False, bottom=False)

    for each_elem in infeasible_instances:
        plt.text(each_elem[0], each_elem[1], "X", horizontalalignment='center',  color='black', weight='semibold', fontsize=40)
   
    
    myplot._legend.remove()
    myplot.savefig(OUTPUT_DIR / Path("fig_5_a.pdf"), bbox_inches = "tight")

## *****raw numbers **********************
    allgather_normal_NDv2_2chassis = pd.DataFrame(list(zip(transfer_size, TECCL_allgather_NDv2_2Chassis_alg_bnw, normal_2)), columns = ['transfer size', 'algorithm bandwidth (Gb/s)', 'experiment'])
    allgather_normal_NDv2_4chassis = pd.DataFrame(list(zip(transfer_size, TECCL_allgather_NDv2_4Chassis_alg_bnw, normal_4)), columns = ['transfer size', 'algorithm bandwidth (Gb/s)', 'experiment'])
    allgather_early_stop_NDv2_2chassis = pd.DataFrame(list(zip(transfer_size, TECCL_allgather_NDv2_2Chassis_alg_bnw_early_stop, early_stop_2)), columns = ['transfer size', 'algorithm bandwidth (Gb/s)', 'experiment'])
    allgather_early_stop_NDv2_4chassis = pd.DataFrame(list(zip(transfer_size, TECCL_allgather_NDv2_4Chassis_alg_bnw_early_stop, early_stop_4)), columns = ['transfer size', 'algorithm bandwidth (Gb/s)', 'experiment'])
    alltoall_normal_NDv2_2chassis = pd.DataFrame(list(zip(transfer_size, TECCL_NDv2_2_chassis_alltoall_bw, normal_2_alltoall)), columns = ['transfer size', 'algorithm bandwidth (Gb/s)', 'experiment'])
    alltoall_normal_NDv2_4chassis = pd.DataFrame(list(zip(transfer_size, TECCL_NDv2_4_chassis_alltoall_bw, normal_4_alltoall)), columns = ['transfer size', 'algorithm bandwidth (Gb/s)', 'experiment'])
    all_data_allgather_NDv2 = pd.concat([allgather_normal_NDv2_2chassis, allgather_normal_NDv2_4chassis, allgather_early_stop_NDv2_2chassis, allgather_early_stop_NDv2_4chassis, alltoall_normal_NDv2_2chassis, alltoall_normal_NDv2_4chassis ])
    all_data_allgather_NDv2 = all_data_allgather_NDv2.reset_index()
    


    ##******************Solver speedup*********************
    
    TECCL_solver_time_ndv2_2chassis_normal_allgather = [7201.048, 7214.155, 7209.458, 7208.702, 152.604, 160.101, 59.545, 27.607, 18.795, 12.261, 50.282]
    
    TECCL_solver_time_ndv2_2chassis_early_stop_allgather = [2.661, 2.371, 2.446, 2.424, 2.402, 4.318, 2.832, 3.935, 12.983, 10.169, 42.941]
    
    TACCL_solver_time_ndv2_2chassis = [7.01055, 6.55721, 8.26914, 2000.73756, 8.36668, 62.65401, 11.16676, 3.66477, 6.33559, 4.29815, 3.01664]

    
    TECCL_solver_time_ndv2_2chassis_alltoall = [336.501, 307.333, 339.922, 280.821, 165.627, 189.469, 218.497, 161.994, 182.077, 69.577, 196.721]
    
    TACCL_solver_time_ndv2_2chassis_alltoall = [1214.69166, 1217.55693, 1220.6024, 1213.90291, 1214.50688, 1213.52101, 1221.77875, 860.87501, 86.03384, 31.13943, 27.65954]

    speedup_solver_time_ndv2_2chassis_alltoall = []
    speedup_solver_time_ndv2_2chassis_normal_allgather = []
    speedup_solver_time_ndv2_2chassis_early_stop_allgather = []
    normal_2_allgather = []
    speedup_2_allgather = []
    normal_2_alltoall = []
    topology_2_allgather_normal = []
    topology_2_allgather_early_stop = []
    topology_2_alltoall = []
    for i in range(len(transfer_size)):
        speedup_solver_time_ndv2_2chassis_normal_allgather += [(TACCL_solver_time_ndv2_2chassis[i] - TECCL_solver_time_ndv2_2chassis_normal_allgather[i]) * 100/ TACCL_solver_time_ndv2_2chassis[i]]
        speedup_solver_time_ndv2_2chassis_early_stop_allgather += [(TACCL_solver_time_ndv2_2chassis[i] - TECCL_solver_time_ndv2_2chassis_early_stop_allgather[i] ) * 100/TACCL_solver_time_ndv2_2chassis[i]]
        speedup_solver_time_ndv2_2chassis_alltoall += [(TACCL_solver_time_ndv2_2chassis_alltoall[i] - TECCL_solver_time_ndv2_2chassis_alltoall[i]) * 100/TACCL_solver_time_ndv2_2chassis_alltoall[i]]
        normal_2_allgather += ['2 ch, AG, opt']
        speedup_2_allgather += ['2 ch, AG, ES']
        normal_2_alltoall += ['2 ch, AtoA']
        topology_2_allgather_normal += ['2 ch\n AG\n opt']
        topology_2_allgather_early_stop += ['2 ch\n AG\n ES']
        topology_2_alltoall += ['2 ch\n AtoA']

    
    TECCL_solver_time_ndv2_4chassis_normal_allgather = [7215.602, 7215.006, 7215.011, 7214.384, 7216.438, 7214.745, 7214.429, 7220.283, 7261.858, 168.072, 168.072]
    
    TECCL_solver_time_ndv2_4chassis_early_stop_allgahter = [7210.569, 7210.166, 7209.849, 7211.204, 7209.673, 640.561, 1496.864, 150.105, 32.443, 129.669, 656.688]
    
    TACCL_solver_time_ndv2_4chassis_allgather = [0, 58.347, 43.69632, 49.17196, 40.43816, 36.15542, 43.96712, 63.60514, 35.5709, 75.8672, 59.8469]

    
    TECCL_solver_time_ndv2_4chassis_alltoall = [392.82, 442.609, 420.456, 412.924, 341.014, 376.504, 504.149, 362.383, 2083.671, 976.711, 710.112]
    
    TACCL_solver_time_ndv2_4chassis_alltoall = [1400.85542, 1401.2642, 1430.46246, 1391.72074, 1397.45292, 0, 1418.8379, 732.03262, 979.89115, 947.03422, 901.50597]

    speedup_solver_time_ndv2_4chassis_normal_allgather = []
    speedup_solver_time_ndv2_4chassis_early_stop_allgahter = []
    speedup_solver_time_ndv2_4chassis_alltoall = []
    normal_4_allgather = []
    early_stop_4_allgather = []
    normal_4_alltoall = []
    topology_4_allgahter_normal = []
    topology_4_allgather_early_stop =[]
    topology_4_alltoall = []
    for i in range(len(transfer_size)):
        if TACCL_solver_time_ndv2_4chassis_allgather[i] != 0:
            speedup_solver_time_ndv2_4chassis_normal_allgather += [(TACCL_solver_time_ndv2_4chassis_allgather[i] - TECCL_solver_time_ndv2_4chassis_normal_allgather[i] ) * 100/ TACCL_solver_time_ndv2_4chassis_allgather[i]]
            speedup_solver_time_ndv2_4chassis_early_stop_allgahter += [(TACCL_solver_time_ndv2_4chassis_allgather[i] - TECCL_solver_time_ndv2_4chassis_early_stop_allgahter[i] ) * 100/ TACCL_solver_time_ndv2_4chassis_allgather[i]]
        else: 
            speedup_solver_time_ndv2_4chassis_normal_allgather += [100]
            speedup_solver_time_ndv2_4chassis_early_stop_allgahter += [100]
        if TACCL_solver_time_ndv2_4chassis_alltoall[i] != 0:
            speedup_solver_time_ndv2_4chassis_alltoall += [(TACCL_solver_time_ndv2_4chassis_alltoall[i] - TECCL_solver_time_ndv2_4chassis_alltoall[i]) * 100 / TACCL_solver_time_ndv2_4chassis_alltoall[i]]
        else:
            speedup_solver_time_ndv2_4chassis_alltoall += [100]
        normal_4_alltoall += ['4 ch, AtoA']
        normal_4_allgather += ['4 ch, AG, opt']
        early_stop_4_allgather += ['4ch, AG, ES']
        topology_4_allgahter_normal += ['4 ch\n AG\n opt']
        topology_4_allgather_early_stop += ['4 ch\n AG\n ES']
        topology_4_alltoall += ['4 ch\n AtoA']

    speedup_solver_time_ndv2_4chassis_normal_allgather_pd = pd.DataFrame(list(zip(transfer_size, speedup_solver_time_ndv2_4chassis_normal_allgather, normal_4_allgather, topology_4_allgahter_normal)), columns = ['transfer size', 'solver time improvement', 'experiment', 'topology'])
    speedup_solver_time_ndv2_4chassis_early_stop_allgather_pd = pd.DataFrame(list(zip(transfer_size, speedup_solver_time_ndv2_4chassis_early_stop_allgahter, early_stop_4_allgather, topology_4_allgather_early_stop)), columns = ['transfer size', 'solver time improvement', 'experiment', 'topology'])
    speedup_solver_time_ndv2_4chassis_alltoall_pd = pd.DataFrame(list(zip(transfer_size, speedup_solver_time_ndv2_4chassis_alltoall, normal_4_alltoall, topology_4_alltoall)), columns = ['transfer size', 'solver time improvement', 'experiment', 'topology'])
    speedup_solver_time_ndv2_2chassis_normal_allgather_pd = pd.DataFrame(list(zip(transfer_size, speedup_solver_time_ndv2_2chassis_normal_allgather, normal_2_allgather, topology_2_allgather_normal )), columns = ['transfer size', 'solver time improvement', 'experiment', 'topology'])
    speedup_solver_time_ndv2_2chassis_early_stop_allgather_pd = pd.DataFrame(list(zip(transfer_size, speedup_solver_time_ndv2_2chassis_early_stop_allgather, early_stop_2, topology_2_allgather_early_stop)), columns = ['transfer size', 'solver time improvement', 'experiment', 'topology'])
    speedup_solver_time_ndv2_2chassis_alltoall_pd = pd.DataFrame(list(zip(transfer_size, speedup_solver_time_ndv2_2chassis_alltoall, normal_2_alltoall, topology_2_alltoall)), columns = ['transfer size', 'solver time improvement', 'experiment', 'topology'])
    


    all_ndv2_solver_time_speedups = pd.concat([speedup_solver_time_ndv2_2chassis_early_stop_allgather_pd, speedup_solver_time_ndv2_2chassis_normal_allgather_pd, speedup_solver_time_ndv2_2chassis_alltoall_pd, speedup_solver_time_ndv2_4chassis_early_stop_allgather_pd, speedup_solver_time_ndv2_4chassis_normal_allgather_pd, speedup_solver_time_ndv2_4chassis_alltoall_pd])
    all_ndv2_solver_time_speedups = all_ndv2_solver_time_speedups.reset_index()


    plt.figure().clear()
    myplot = sns.violinplot(x='topology', y='solver time improvement', 
                data=all_ndv2_solver_time_speedups,   palette="pastel",  height=15, aspect=11.7/8.27, cut=0)
  
    
    plt.xlabel(" # chassis, demand type", fontsize=20)
    plt.ylabel('Improvement in solver time (%)', fontsize=17)
    sns.despine(left=False, right =False, top=False, bottom=False)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    sns.despine(left=False, right =False, top=False, bottom=False)
    myplot.set(yscale="log")
    fig = plt.gcf()
    file_name = str(OUTPUT_DIR / Path("fig_6_a.pdf"))
    fig.savefig(file_name, bbox_inches = "tight")


    ##################################################DGX############################################################
    
    TEECL_dgx_allgather_bw = read_alg_bnw("DGX2_2_chassis_AllGather_Fast.csv")
    
    TECCL_dgx_allgather_bw_early_stop = read_alg_bnw("DGX2_2_chassis_AllGather_Fast_Early_Stop.csv")
    
    TACCL_dgx_allgather_bw = [107.908623, 107.3330259, 105.0730586, 99.75062344, 65.68144499, 31.89792663, 10.15389497, 2.628984555, 0.688705234, 0.172913154, 0.046066497]


    
    TECCL_dgx_alltoall_bw = [12.46105919, 12.46105919, 12.42236025, 12.42236025, 12.26993865, 11.79941003, 10.69518717, 7.366482505, 3.407155026, 0.952380952, 0.246609125]
    
    TACCL_dgx_alltoall_bw = [11.12932611, 11.3865834, 11.0579332, 0, 8.444162972, 4.537205082, 1.730852445, 0.485790624, 0.126968004, 0.030934133, 0.008316389]

    dgx2_speedup_normal = []
    dgx2_speedup_early_stop = []
    dgx2_speedup_normal_alltoall = []
    early_stop = []
    normal = []
    normal_alltoall = []
    infeasible_instances = []
    for i in range(len(TACCL_dgx_allgather_bw)):
        dgx2_speedup_normal += [(TEECL_dgx_allgather_bw[i] - TACCL_dgx_allgather_bw[i]) * 100/TACCL_dgx_allgather_bw[i]]
        dgx2_speedup_early_stop += [(TECCL_dgx_allgather_bw_early_stop[i] - TACCL_dgx_allgather_bw[i]) * 100 / TACCL_dgx_allgather_bw[i]]
        if TACCL_dgx_alltoall_bw[i] != 0:
            dgx2_speedup_normal_alltoall += [(TECCL_dgx_alltoall_bw[i] - TACCL_dgx_alltoall_bw[i]) * 100/ TACCL_dgx_alltoall_bw[i]]
            normal_alltoall += ['2 ch, AtoA']
        else:
            infeasible_instances += [(transfer_size[i], dgx2_speedup_normal_alltoall[-1])]
            dgx2_speedup_normal_alltoall += [dgx2_speedup_normal_alltoall[-1]]
            normal_alltoall += ['2 ch, AtoA']
        early_stop += ['2 ch, AG, ES']
        normal += ['2 ch, AG, opt']

   

    speedup_dgx2_normal = pd.DataFrame(list(zip(transfer_size, dgx2_speedup_normal, normal)), columns = ['transfer size', 'perf improvement', 'experiment'])
    speedup_dgx2_early_stop = pd.DataFrame(list(zip(transfer_size, dgx2_speedup_early_stop, early_stop)), columns = ['transfer size', 'perf improvement', 'experiment'])
    speedup_dgx2_alltoall = pd.DataFrame(list(zip(transfer_size, dgx2_speedup_normal_alltoall, normal_alltoall)), columns = ['transfer size', 'perf improvement', 'experiment'])


    dgx_speedup = pd.concat([speedup_dgx2_early_stop, speedup_dgx2_normal, speedup_dgx2_alltoall])
    dgx_speedup = dgx_speedup.reset_index()
    
    myplot = sns.relplot(x='transfer size', y='perf improvement', 
                data=dgx_speedup, hue = 'experiment',  style= 'experiment', kind="line", ci=95, palette="muted",linewidth = 7, height=8, aspect=11.7/8.27)
  
    legend = plt.legend( loc="upper left", frameon=True, fontsize=24)
    for line in legend.get_lines():
        line.set_linewidth(6)
    plt.xlabel("Output buffer size", fontsize=40)
    plt.ylabel("Improvement in algo bandwidth (%)", fontsize=29)
    sns.despine(left=False, right =False, top=False, bottom=False)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    sns.despine(left=False, right =False, top=False, bottom=False)
    for each_elem in infeasible_instances:
        plt.text(each_elem[0], each_elem[1], "X", horizontalalignment='center',  color='black', weight='semibold', fontsize=40)
    
    myplot._legend.remove()
    myplot.savefig(OUTPUT_DIR / Path("fig_5_b.pdf"))


    #*******************solver time ***************************
    
    TECCL_dgx2_solver_time_allgather = [7203.983, 7204.866, 7206.617, 7205.181, 7204.694, 7206.15, 476.409, 129.694, 12.909, 57.357, 204.85]
    
    TECCL_dgx2_solver_time_allgather_early_stop = [270.604, 316.412, 175.027, 269.626, 202.126, 250.323, 24.722, 140.772, 15.046, 67.101, 204.828]
    
    TACCL_dgx2_solver_time_allgather = [8.97775, 120.21598, 281.90938, 52.39926, 9.06847, 94.7856, 8.92938, 231.48384, 10.11165, 69.19717, 46.44718]
    
    TECCL_dgx2_solver_time_alltoall = [1656.349, 1519.48, 1494.933, 1689.501, 1404.961, 902.355, 1965.841, 2290.591, 3133.322, 1134.111, 1288.639]
    
    TACCL_dgx2_solver_time_alltoall = [3324.00053, 3318.09809, 3309.65569, 184.982, 3310.83728, 2126.66984, 2137.59128, 2121.05876, 2119.98814, 2119.50663, 2119.1703]


    speedup_solver_time_dgx2_allgather_normal = []
    speedup_solver_time_dgx2_allgather_early_stop = []
    speedup_solver_time_dgx2_alltoall = []
    normal_dgx2_allgather = []
    early_stop_dgx2_allgather = []
    alltoall_dgx2 =[]
    topology_normal_dgx2_allgather = []
    topology_early_stop_dgx2 = []
    topology_alltoall_dgx2 = []
    for i in range(len(transfer_size)):
        speedup_solver_time_dgx2_allgather_normal += [(TACCL_dgx2_solver_time_allgather[i] - TECCL_dgx2_solver_time_allgather[i]) * 100/TACCL_dgx2_solver_time_allgather[i]]
        speedup_solver_time_dgx2_allgather_early_stop += [(TACCL_dgx2_solver_time_allgather[i] - TECCL_dgx2_solver_time_allgather_early_stop[i]) * 100/TACCL_dgx2_solver_time_allgather[i]]
        speedup_solver_time_dgx2_alltoall += [(TACCL_dgx2_solver_time_alltoall[i] - TECCL_dgx2_solver_time_alltoall[i]) * 100/TACCL_dgx2_solver_time_alltoall[i]]
        normal_dgx2_allgather += ['2 ch, AG, opt']
        early_stop_dgx2_allgather += ['2 ch, AG, ES']
        alltoall_dgx2 += ['2ch, AtoA']
        topology_normal_dgx2_allgather += ['2 ch\n AG\n opt']
        topology_early_stop_dgx2 += ['2 ch\n AG\n ES']
        topology_alltoall_dgx2 += ['2 ch\n AtoA']

    speedup_solver_time_normal_allgather_dgx2 = pd.DataFrame(list(zip(transfer_size, speedup_solver_time_dgx2_allgather_normal, normal_dgx2_allgather, topology_normal_dgx2_allgather)) , columns = ['transfer size', 'solver time improvement', 'experiment', 'topology'])
    speedup_solver_time_early_stop_allgather_dgx2 = pd.DataFrame(list(zip(transfer_size, speedup_solver_time_dgx2_allgather_early_stop, early_stop_dgx2_allgather, topology_early_stop_dgx2)), columns = ['transfer size', 'solver time improvement', 'experiment', 'topology'])
    speedup_solver_time_alltoall_dgx2 = pd.DataFrame(list(zip(transfer_size, speedup_solver_time_dgx2_alltoall, alltoall_dgx2, topology_alltoall_dgx2)), columns = ['transfer size', 'solver time improvement', 'experiment', 'topology'])
    

    speedup_dgx2_solver_time = pd.concat([speedup_solver_time_normal_allgather_dgx2, speedup_solver_time_early_stop_allgather_dgx2, speedup_solver_time_alltoall_dgx2])
    speedup_dgx2_solver_time = speedup_dgx2_solver_time.reset_index()

    

    
    plt.figure().clear()
    myplot = sns.violinplot(x='topology', y='solver time improvement', 
                data=speedup_dgx2_solver_time,   palette="pastel",  height=15, aspect=11.7/8.27, cut=0)
  
    
    plt.xlabel(" # chassis, demand type", fontsize=20)
    plt.ylabel('Improvement in solver time (%)', fontsize=17)
    sns.despine(left=False, right =False, top=False, bottom=False)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    sns.despine(left=False, right =False, top=False, bottom=False)
    myplot.set(yscale="log")
    fig = plt.gcf()
    file_name = str(OUTPUT_DIR / Path("fig_6_b.pdf"))
    fig.savefig(file_name, bbox_inches = "tight")

def plot_small_vs_large_epochs():
    
    ndv2_min_2chassis_solver_time_allgather = [7201.048, 7214.155, 7209.458, 7208.702, 152.604, 160.101, 59.545, 27.607, 18.795, 12.261, 50.282]
    
    ndv2_min_2chassis_solution_time_allgather = read_collective_time("NDv2_2_chassis_AllGather_Fast.csv")
    
    ndv2_max_2chassis_solution_time_allgather = read_collective_time("NDv2_2_chassis_AllGather_Slow.csv")
    
    ndv2_max_2chassis_solver_time_allgather = [0.944, 0.773, 0.777, 0.774, 0.769, 1.043, 1.09, 1.742, 3.349, 21.562, 89.066]


    
    ndv2_2chassis_allgather = pd.DataFrame(list(zip(ndv2_min_2chassis_solution_time_allgather, ndv2_min_2chassis_solver_time_allgather, ndv2_max_2chassis_solution_time_allgather, ndv2_max_2chassis_solver_time_allgather)),
                                        columns = [' total time without alpha_min', ' solver time_min', ' total time without alpha_max', ' solver time_max'])
    ndv2_2chassis_allgather['experiment'] = 'NDv2\n  AG'

    
    ndv2_min_2chassis_solution_time_alltoall = read_collective_time("NDv2_2_chassis_AlltoAll_Fast.csv")
    
    ndv2_min_2chassis_solver_time_alltoall = [336.501, 307.333, 339.922, 280.821, 165.627, 189.469, 218.497, 161.994, 182.077, 69.577, 196.721]
    
    ndv2_max_2chassis_solver_time_alltoall = [14.815, 14.357, 11.009, 9.958, 11.812, 10.845, 9.974, 10.461, 8.827, 20.901, 276.47]
    
    ndv2_max_2chassis_solution_time_alltoall = read_collective_time("NDv2_2_chassis_AlltoAll_Slow.csv")

    ndv2_2chassis_alltoall = pd.DataFrame(list(zip(ndv2_min_2chassis_solution_time_alltoall, ndv2_min_2chassis_solver_time_alltoall, ndv2_max_2chassis_solution_time_alltoall, ndv2_max_2chassis_solver_time_alltoall)),
                                          columns = [' total time without alpha_min', ' solver time_min', ' total time without alpha_max', ' solver time_max'])
    ndv2_2chassis_alltoall['experiment'] = 'NDv2\n  AtoA'

    
    dgx2_min_solver_time_allgather = [7203.983, 7204.866, 7206.617, 7205.181, 7204.694, 7206.15, 476.409, 129.694, 12.909, 57.357, 204.85]
    
    dgx2_min_solution_time_allgather = read_collective_time("DGX2_2_chassis_AllGather_Fast.csv")
    
    dgx2_max_solution_time_allgather = read_collective_time("DGX2_2_chassis_AllGather_Slow.csv")
    
    dgx2_max_solver_time_allgather = [1.414, 1.275, 1.314, 1.283, 1.226, 1.248, 1.523, 5.206, 17.4, 62.001, 263.334]

    dgx2_allgather = pd.DataFrame(list(zip(dgx2_min_solution_time_allgather, dgx2_min_solver_time_allgather, dgx2_max_solution_time_allgather, dgx2_max_solver_time_allgather)),
                                columns = [' total time without alpha_min', ' solver time_min', ' total time without alpha_max', ' solver time_max'])
    dgx2_allgather['experiment'] = 'DGX2 \n AG'

    
    dgx2_min_solution_time_alltoall = [80250, 20544, 5152, 1288, 326, 84.75, 23.936, 8.688, 4.696, 4.2, 4.055]
    
    dgx2_min_solver_time_alltoall = [1656.349, 1519.48, 1494.933, 1689.501, 1404.961, 902.355, 1965.841, 2290.591, 3133.322, 1134.111, 1288.639]
    
    dgx2_max_solution_time_alltoall = [82500, 21120, 5280, 1320, 340, 92.5, 25.6, 8.96, 4.56, 4.2, 4.055]
    
    dgx2_max_solver_time_alltoall =[34.907, 34.469, 34.145, 35.185, 34.457, 35.318, 38.321, 47.518, 80.37, 569.405, 1288.639]

    dgx2_alltoall = pd.DataFrame(list(zip(dgx2_min_solution_time_alltoall, dgx2_min_solver_time_alltoall, dgx2_max_solution_time_alltoall, dgx2_max_solver_time_alltoall)),
                                columns = [' total time without alpha_min', ' solver time_min', ' total time without alpha_max', ' solver time_max'])
    dgx2_alltoall['experiment'] = 'DGX2\n  AtoA'

    all_numbers = pd.concat([ndv2_2chassis_allgather, ndv2_2chassis_alltoall, dgx2_allgather, dgx2_alltoall])
    all_numbers = all_numbers.reset_index()
    all_numbers['speedup solver time'] = all_numbers.apply(lambda row: (row[' solver time_min'] - row[' solver time_max'])*100/row[' solver time_max'], axis = 1)
    all_numbers['speedup transfer time'] = all_numbers.apply(lambda row: (row[' total time without alpha_min'] - row[' total time without alpha_max']) * 100/row[' total time without alpha_max'], axis = 1)
    all_numbers['speedup transfer time'] = all_numbers['speedup transfer time'].apply(lambda x: x if abs(x)> 1e-6 else 0 )

    print(all_numbers[all_numbers['speedup solver time'] < 0].shape[0]/all_numbers.shape[0])

    plt.figure().clear()
    myplot = sns.violinplot(x='experiment', y='speedup solver time', 
                data=all_numbers,   palette="pastel",  height=18, aspect=11.7/8.27)
  
    
    plt.xlabel("topology,  demand", fontsize=24)
    plt.ylabel('Difference in solver time (%)', fontsize=19)
    sns.despine(left=False, right =False, top=False, bottom=False)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    sns.despine(left=False, right =False, top=False, bottom=False)
    myplot.set(yscale="log")
    fig = plt.gcf()
    file_name = str(OUTPUT_DIR / Path("fig_10_a.pdf"))
    fig.savefig(file_name, bbox_inches = "tight")

    plt.figure().clear()
    myplot = sns.violinplot(x='experiment', y='speedup transfer time', 
                data=all_numbers,   palette="pastel",  height=18, aspect=11.7/8.27)
  
    
    plt.xlabel("topology,  demand", fontsize=24)
    plt.ylabel('Difference in transfer (%)', fontsize=21)
    sns.despine(left=False, right =False, top=False, bottom=False)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    sns.despine(left=False, right =False, top=False, bottom=False)
    fig = plt.gcf()
    file_name = str(OUTPUT_DIR / Path("fig_10_b.pdf"))
    fig.savefig(file_name, bbox_inches = "tight")

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_taccl_comparison()
    plot_small_vs_large_epochs()