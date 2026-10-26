from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import pyvale
import valmetrics as vm

def main() -> None:
    print(80*"=")
    print("PULSE 25X: Thermocouple MAVM analysis")
    print(80*"=")

    EXP_DIR = Path.cwd() / "STC_Exp_TCs_25X"
    SIM_DIR = Path.cwd() / "STC_ProbSim_FieldsFull_25X_v4"

    sens_ax_labels = (r"Temp. [$^{\circ}C$]",)*10 + (r"Coil RMS Voltage [$V$]",)
    sens_tags = ("Temp",)*10 + ("Volts",)
    sens_num = len(sens_tags)

    save_path = Path.cwd() / "images_pointsensors_pulse25X_v4"

    if not save_path.is_dir():
        save_path.mkdir(exist_ok=True,parents=True)

    plot_opts = pyvale.sensorsim.PlotOptsGeneral()

    fig_ind: int = 0
    exp_c: str = "tab:orange"
    sim_c: str = "tab:blue"

    ax_lims = {}
    
    #---------------------------------------------------------------------------
    # Simulation: Load Data
    print(f"Loading simulation data from:\n    {SIM_DIR}")

    # Reduced: 5000 = 100 aleatory x 50 epistemic
    # Full: 400 aleatory x 250 epistemic
    # exp_data = exp_data.reshape(samps_n,epis_n,alea_n)
    #samps_n: int = 5000
    epis_n: int = 250#50
    alea_n: int = 400#100

    sim_keys = {"TC1":0,
                "TC2":1,
                "TC3":2,
                "TC4":3,
                "TC5":4,
                "TC6":5,
                "TC7":6,
                "TC8":7,
                "TC9":8,
                "TC10":9,
                "CV":10,}

    sim_path = SIM_DIR / "SamplingResultsOnlyPointSensors.csv"
    sim_data_arr = pd.read_csv(sim_path).to_numpy()

    sim_data = {}
    for kk in sim_keys:
        sim_data[kk] = sim_data_arr[:,sim_keys[kk]]

        sim_data[kk] = sim_data[kk].reshape(epis_n,alea_n)

        # print(kk)
        # print(f"{sim_data[kk].shape=}")
        # print()

    #---------------------------------------------------------------------------
    # Experiment: Load Data
    print(f"Loading experimental data from:\n    {EXP_DIR}")

    exp_file_pre = ["Pulse253","Pulse254","Pulse255"]
    exp_file_post = ["SteadyDICData","SteadyHIVEData","SteadyPICOData"]
    exp_file_slices = [slice(2,11),slice(2,3),slice(1,2)]
    # NOTE: actually in file = 3,5,6,8,9,10
    exp_data_keys = [["TC1","TC3","TC4","TC5","TC6","TC7","TC8","TC9","TC10"],
                     ["TC2"],
                     ["CV"]]

    exp_data = {}
    for pp in exp_file_pre:
        for ii,ss in enumerate(exp_file_post):
            file = EXP_DIR / f"{pp}_{ss}.csv"

            data = pd.read_csv(file)
            data = data.to_numpy()
            data = data[:,exp_file_slices[ii]]

            for jj,kk in enumerate(exp_data_keys[ii]):
                exp_data[kk] = data[:,jj]

    print("Loading experimental epistemic error data.")
    exp_epis_file = Path.cwd() / "STC_ProbSim_MetaData_25X" / "STC2_1550A_SteadySummary.csv"
    exp_epis_err = pd.read_csv(exp_epis_file)
    exp_epis_err = exp_epis_err.to_numpy()
    # Extract the last row and columns where the 10 TCs and coil voltage are
    exp_epis_err = np.array(exp_epis_err[-1,-11:],dtype=np.float64)

    # Move the coil voltage from the start to end to match everything else
    temp = np.copy(exp_epis_err)
    exp_epis_err[-1] = temp[0]
    exp_epis_err[0:-1] = temp[1:]
    del temp

    #---------------------------------------------------------------------------
    # Simulation: Find Min/Max CDF over all eps
    print()
    print(80*"-")
    print("Extracting max and min cdf over all simulations.")

    # Extract all the simulation cdfs by looping over the epistemic sampling
    sim_cdfs_all = {}
    sim_data_lims = {}
    sim_cdfs_lims ={}

    for kk in sim_data:
        cdfs = []

        max_data = sim_data[kk][0,:]
        min_data = sim_data[kk][0,:]

        min_cdf = stats.ecdf(sim_data[kk][0,:]).cdf
        max_cdf = stats.ecdf(sim_data[kk][0,:]).cdf

        sum_min_cdf = np.sum(min_cdf.quantiles)
        sum_max_cdf = np.sum(max_cdf.quantiles)

        accum_cdf = np.zeros_like(max_cdf.quantiles)

        for ee in range(epis_n):
            this_data = sim_data[kk][ee,:]
            this_cdf = stats.ecdf(sim_data[kk][ee,:]).cdf
            this_cdf_sum = np.sum(this_cdf.quantiles)

            if len(accum_cdf) != len(this_cdf.quantiles):
                accum_cdf = accum_cdf + np.sort(this_data)
            else:
                accum_cdf = accum_cdf + this_cdf.quantiles

            if this_cdf_sum > sum_max_cdf:
                sum_max_cdf = this_cdf_sum
                max_cdf = this_cdf
                max_data = sim_data[kk][ee,:]

            if this_cdf_sum < sum_min_cdf:
                sum_min_cdf = this_cdf_sum
                min_cdf = this_cdf
                min_data = sim_data[kk][ee,:]

            cdfs.append(this_cdf)

        sim_cdfs_all[kk] = cdfs

        mean_data = accum_cdf / epis_n
        this_data = {}
        this_data["max"] = max_data
        this_data["min"] = min_data
        this_data["nom"] = mean_data
        sim_data_lims[kk] = this_data

        mean_cdf = stats.ecdf(mean_data).cdf
        this_cdf = {}
        this_cdf["max"] = max_cdf
        this_cdf["min"] = min_cdf
        this_cdf["nom"] = mean_cdf
        sim_cdfs_lims[kk] = this_cdf


    PLOT_ALL_SIM_CDFS = True

    if PLOT_ALL_SIM_CDFS:
        for ii,kk in enumerate(sim_cdfs_all):
            fig, axs=plt.subplots(1,1,
                                figsize=plot_opts.single_fig_size_landscape,
                                layout="constrained")
            fig.set_dpi(plot_opts.resolution)

            for cc in sim_cdfs_all[kk]:
                axs.ecdf(cc.quantiles,color='tab:blue',linewidth=plot_opts.lw)

            axs.ecdf(sim_cdfs_lims[kk]["max"].quantiles
                     ,ls="--",color="black",linewidth=plot_opts.lw)
            axs.ecdf(sim_cdfs_lims[kk]["min"].quantiles
                     ,ls="--",color="black",linewidth=plot_opts.lw)
            axs.ecdf(sim_cdfs_lims[kk]["nom"].quantiles
                     ,ls="-",color="black",linewidth=plot_opts.lw)

            axs.set_title(kk,fontsize=plot_opts.font_head_size)
            axs.set_xlabel(sens_ax_labels[ii],fontsize=plot_opts.font_ax_size)
            axs.set_ylabel("Probability",fontsize=plot_opts.font_ax_size)

            fig_name = f"{fig_ind}_sim_allcdfs_{kk}.png"
            ax_lims[fig_name] = axs.get_xlim()
            save_fig_path = save_path / fig_name
            fig.savefig(save_fig_path,dpi=300,format="png",bbox_inches="tight")

    #---------------------------------------------------------------------------
    # Experiment: Calculate min/max cdf based on epistemic error

    exp_data_lims = {}
    exp_cdfs_lims = {}
    for ii,kk in enumerate(exp_data):
        this_exp = {}
        this_exp["nom"] = exp_data[kk]
        this_exp["min"] = exp_data[kk] - exp_epis_err[ii]
        this_exp["max"] = exp_data[kk] + exp_epis_err[ii]
        exp_data_lims[kk] = this_exp

        this_cdf = {}
        if np.any(np.isnan(exp_data_lims[kk]["nom"])):
            this_cdf["nom"] = None
            this_cdf["min"] = None
            this_cdf["max"] = None

        else:
            this_cdf["nom"] = stats.ecdf(exp_data[kk]).cdf
            this_cdf["min"] = stats.ecdf(exp_data[kk] - exp_epis_err[ii]).cdf
            this_cdf["max"] = stats.ecdf(exp_data[kk] + exp_epis_err[ii]).cdf
        exp_cdfs_lims[kk] = this_cdf

    #---------------------------------------------------------------------------
    # Calculate MAVM
    # PLOT_MAVM_INDIVIDUAL = False

    mavm = {}
    dplus_max = {}
    dminus_max = {}
    mavm_keys = ("min","max")

    print(80*"-")
    print("Calculating MAVM...")
    for ii,kk in enumerate(exp_data_lims): # Loop over sensors: TCs + CV

        sim_cdf_sum_upper = None
        sim_cdf_sum_lower = None
        this_mavm = {}

        for es in mavm_keys: # Loop over EXP stats: min,max
            for ss in mavm_keys: # Loop over SIM stats: min,max

                # Some TCs missing from exp data, skip them
                if not np.any(np.isnan(exp_data_lims[kk][es])):

                    print(80*"-")
                    print(f"{kk=}")
                    print(f"sens-{kk}_sim-{ss}_exp-{es}")
                    print(f"{exp_data_lims[kk][es].shape=}")
                    print(f"{sim_data_lims[kk][ss].shape=}")
                    print()
                    print(f"{np.mean(exp_data_lims[kk][es])=}")
                    print(f"{np.mean(sim_data_lims[kk][ss])=}")
                    print()

                    stat_key = f"sim-{ss}_exp-{es}"

                    this_mavm[stat_key] = vm.mavm(sim_data_lims[kk][ss],
                                                  exp_data_lims[kk][es])

                    print(f"{this_mavm[stat_key]['d+']=}")
                    print(f"{this_mavm[stat_key]['d-']=}")
                    print(80*"-")

                    check_upper = np.sum(this_mavm[stat_key]["F_"] 
                                         + this_mavm[stat_key]["d+"])
                    check_lower = np.sum(this_mavm[stat_key]["F_"] 
                                         - this_mavm[stat_key]["d-"])

                    if ss == "max":
                        if ((sim_cdf_sum_upper is None) or 
                            (check_upper > sim_cdf_sum_upper)):
                            sim_cdf_sum_upper = check_upper
                            
                            dplus_max_val = this_mavm[stat_key]["d+"]
                            dplus_max[kk] = this_mavm[stat_key]
                            dplus_max[kk]["stat_key"] = stat_key

                    if ss == "min":
                        if ((sim_cdf_sum_lower is None) or 
                            (check_lower < sim_cdf_sum_lower)):
                            sim_cdf_sum_lower = check_lower
                            
                            dminus_max_val = this_mavm[stat_key]["d-"]
                            dminus_max[kk] = this_mavm[stat_key]
                            dminus_max[kk]["stat_key"] = stat_key

        if not np.any(np.isnan(exp_data_lims[kk][es])):
            mavm[kk] = this_mavm

            print(80*"-")
            print(f"{kk=}")
            print(f"{dminus_max_val=}")
            print(f"{dplus_max_val=}")
            print()
            print(f"{dplus_max[kk]['stat_key']=}")
            print(f"{dminus_max[kk]['stat_key']=}")
            print(80*"-")

    #---------------------------------------------------------------------------
    # FIGURE: exp and sim CDFS with epistemic errors

    plt.close("all")
    
    print(80*"-")
    print("Plotting combined MAVM figures...")
    fig_ind += 1

    for ii,kk in enumerate(exp_data_lims): # Loop over sensors: TCs + CV

        if np.any(np.isnan(exp_data_lims[kk]["nom"])):
            continue

        #-----------------------------------------------------------------------
        fig,axs=plt.subplots(1,1,
                         figsize=plot_opts.single_fig_size_landscape,
                         layout="constrained")
        fig.set_dpi(plot_opts.resolution)

        exp_c = "tab:orange"
        sim_c = "tab:blue"

        axs.ecdf(sim_cdfs_lims[kk]["nom"].quantiles,
                 ls="-",color=sim_c,label="sim. nom.",linewidth=plot_opts.lw)
        axs.ecdf(sim_cdfs_lims[kk]["max"].quantiles,
                 ls="--",color=sim_c,label="sim. lims.",linewidth=plot_opts.lw)
        axs.ecdf(sim_cdfs_lims[kk]["min"].quantiles,
                 ls="--",color=sim_c,linewidth=plot_opts.lw)

        axs.ecdf(exp_cdfs_lims[kk]["nom"].quantiles,
                 ls="-",color=exp_c,label="exp. nom.",linewidth=plot_opts.lw)
        axs.ecdf(exp_cdfs_lims[kk]["max"].quantiles,
                 ls="--",color=exp_c,label="exp. lims.",linewidth=plot_opts.lw)
        axs.ecdf(exp_cdfs_lims[kk]["min"].quantiles,
                 ls="--",color=exp_c,linewidth=plot_opts.lw)

        axs.fill_betweenx(sim_cdfs_lims[kk]["nom"].probabilities,
                         sim_cdfs_lims[kk]["min"].quantiles,
                         sim_cdfs_lims[kk]["max"].quantiles,
                         color=sim_c,
                         alpha=0.2)

        axs.fill_betweenx(exp_cdfs_lims[kk]["nom"].probabilities,
                         exp_cdfs_lims[kk]["min"].quantiles,
                         exp_cdfs_lims[kk]["max"].quantiles,
                         color=exp_c,
                         alpha=0.2)

        #axs.legend(loc="upper left",fontsize=6)
        axs.set_title(kk,fontsize=plot_opts.font_head_size)
        axs.set_xlabel(sens_ax_labels[ii],fontsize=plot_opts.font_ax_size)
        axs.set_ylabel("Probability",fontsize=plot_opts.font_ax_size)

        save_fig_path = save_path / f"{fig_ind}_exp-sim-cdfs_{kk}.png"
        fig.savefig(save_fig_path,dpi=300,format="png",bbox_inches="tight")

    #---------------------------------------------------------------------------
    # FIGURE: exp and sim CDFS with epistemic errors with d+/- extremes
    fig_ind += 1
    for ii,kk in enumerate(exp_data_lims): # Loop over sensors: TCs + CV

        if np.any(np.isnan(exp_data_lims[kk]["nom"])):
            continue

        #-----------------------------------------------------------------------
        fig,axs=plt.subplots(1,1,
                         figsize=plot_opts.single_fig_size_landscape,
                         layout="constrained")
        fig.set_dpi(plot_opts.resolution)

        axs.ecdf(sim_cdfs_lims[kk]["nom"].quantiles,
                 ls="-",color=sim_c,label="sim. nom.",linewidth=plot_opts.lw)
        axs.ecdf(sim_cdfs_lims[kk]["max"].quantiles,
                 ls=":",color=sim_c,label="sim. lims.",linewidth=plot_opts.lw)
        axs.ecdf(sim_cdfs_lims[kk]["min"].quantiles,
                 ls=":",color=sim_c,linewidth=plot_opts.lw)

        axs.ecdf(exp_cdfs_lims[kk]["nom"].quantiles,
                 ls="-",color=exp_c,label="exp. nom.",linewidth=plot_opts.lw)
        axs.ecdf(exp_cdfs_lims[kk]["max"].quantiles,
                 ls=":",color=exp_c,label="exp. lims.",linewidth=plot_opts.lw)
        axs.ecdf(exp_cdfs_lims[kk]["min"].quantiles,
                 ls=":",color=exp_c,linewidth=plot_opts.lw)

        axs.fill_betweenx(sim_cdfs_lims[kk]["nom"].probabilities,
                         sim_cdfs_lims[kk]["min"].quantiles,
                         sim_cdfs_lims[kk]["max"].quantiles,
                         color=sim_c,
                         alpha=0.2,
                         ls=":")

        axs.fill_betweenx(exp_cdfs_lims[kk]["nom"].probabilities,
                         exp_cdfs_lims[kk]["min"].quantiles,
                         exp_cdfs_lims[kk]["max"].quantiles,
                         color=exp_c,
                         alpha=0.2,
                         ls=":")

        dp_c = "tab:green"
        axs.plot(dplus_max[kk]["F_"] + dplus_max[kk]["d+"],
                 dplus_max[kk]["F_Y"],
                 ls="--",linewidth=plot_opts.lw*1.2, label="d+ max.",
                 color=dp_c)
        axs.plot(dplus_max[kk]["F_"] - dplus_max[kk]["d-"],
                 dplus_max[kk]["F_Y"],
                 ls="--",linewidth=plot_opts.lw*1.2,
                 color=dp_c)

        dm_c = "tab:red"
        axs.plot(dminus_max[kk]["F_"] + dminus_max[kk]["d+"],
                 dminus_max[kk]["F_Y"],
                 ls="--",linewidth=plot_opts.lw*1.2, label="d- max.",
                 color=dm_c)
        axs.plot(dminus_max[kk]["F_"] - dminus_max[kk]["d-"],
                 dminus_max[kk]["F_Y"],
                 ls="--",linewidth=plot_opts.lw*1.2,
                 color=dm_c)

        print(80*"-")
        print(f"{kk=}")
        print("HELLO")
        print(f"{dplus_max[kk]['d+']=}")
        print(f"{dminus_max[kk]['d-']=}")
        print(80*"-")

        #axs.legend(loc="upper left",fontsize=6)
        axs.set_title(kk,fontsize=plot_opts.font_head_size)
        axs.set_xlabel(sens_ax_labels[ii],fontsize=plot_opts.font_ax_size)
        axs.set_ylabel("Probability",fontsize=plot_opts.font_ax_size)

        fig_name = f"{fig_ind}_dextremes_wcdfs_{kk}.png"
        ax_lims[fig_name] = axs.get_xlim()
        save_fig_path = save_path / fig_name
        fig.savefig(save_fig_path,dpi=300,format="png",bbox_inches="tight")

    
    plt.close("all")
    #---------------------------------------------------------------------------
    # FIGURES: cleaner d+/- extremes
    fig_ind += 1
    for ii,kk in enumerate(exp_data_lims): # Loop over sensors: TCs + CV

        if np.any(np.isnan(exp_data_lims[kk]["nom"])):
            continue

        #-----------------------------------------------------------------------
        # FIG 1: d outer limits with exp shown
        fig,axs=plt.subplots(1,1,
                         figsize=plot_opts.single_fig_size_landscape,
                         layout="constrained")
        fig.set_dpi(plot_opts.resolution)

        axs.ecdf(sim_cdfs_lims[kk]["nom"].quantiles,
                 ls="-",color=sim_c,label="sim. nom.",linewidth=plot_opts.lw)
        axs.ecdf(sim_cdfs_lims[kk]["max"].quantiles,
                 ls=":",color=sim_c,label="sim. lims.",linewidth=plot_opts.lw)
        axs.ecdf(sim_cdfs_lims[kk]["min"].quantiles,
                 ls=":",color=sim_c,linewidth=plot_opts.lw)

        axs.ecdf(exp_cdfs_lims[kk]["nom"].quantiles,
                 ls="-",color=exp_c,label="exp. nom.",linewidth=plot_opts.lw)
        axs.ecdf(exp_cdfs_lims[kk]["max"].quantiles,
                 ls=":",color=exp_c,label="exp. lims.",linewidth=plot_opts.lw)
        axs.ecdf(exp_cdfs_lims[kk]["min"].quantiles,
                 ls=":",color=exp_c,linewidth=plot_opts.lw)

        axs.fill_betweenx(sim_cdfs_lims[kk]["nom"].probabilities,
                         sim_cdfs_lims[kk]["min"].quantiles,
                         sim_cdfs_lims[kk]["max"].quantiles,
                         color=sim_c,
                         alpha=0.2,
                         ls=":")

        axs.fill_betweenx(exp_cdfs_lims[kk]["nom"].probabilities,
                         exp_cdfs_lims[kk]["min"].quantiles,
                         exp_cdfs_lims[kk]["max"].quantiles,
                         color=exp_c,
                         alpha=0.2,
                         ls=":")

        dplus_plot = dplus_max[kk]
        dminus_plot = dminus_max[kk]
        axs.fill_betweenx(dplus_plot["F_Y"],
                          dminus_plot["F_"] - dminus_plot["d-"],
                          dplus_plot["F_"] + dplus_plot["d+"],
                         color="black",
                         alpha=0.2,
                         ls=":")

        axs.ecdf(sim_cdfs_lims[kk]["max"].quantiles,
                 ls="-",color=sim_c,label="sim. lims.",linewidth=plot_opts.lw)
        axs.ecdf(sim_cdfs_lims[kk]["min"].quantiles,
                 ls="-",color=sim_c,linewidth=plot_opts.lw)

        axs.fill_betweenx(sim_cdfs_lims[kk]["nom"].probabilities,
                         sim_cdfs_lims[kk]["min"].quantiles,
                         sim_cdfs_lims[kk]["max"].quantiles,
                         color=sim_c,
                         alpha=0.2,
                         ls=":")

        dm_c = "black"
        axs.plot(dplus_plot["F_"] + dplus_plot["d+"],
                 dplus_plot["F_Y"],
                 ls=":",linewidth=plot_opts.lw*1.2, label="d+",
                 color=dm_c)
        axs.plot(dminus_plot["F_"] - dminus_plot["d-"],
                 dminus_plot["F_Y"],
                 ls="--",linewidth=plot_opts.lw*1.2, label="d-",
                 color=dm_c)


        title_str = f"{kk}"
        #axs.legend(loc="upper left",fontsize=6)
        axs.set_title(title_str,fontsize=plot_opts.font_head_size)
        axs.set_xlabel(sens_ax_labels[ii],fontsize=plot_opts.font_ax_size)
        axs.set_ylabel("Probability",fontsize=plot_opts.font_ax_size)

        fig_name = f"{fig_ind}_dextremes_wexpcdfs_{kk}.png"
        ax_lims[fig_name] = axs.get_xlim()
        save_fig_path = save_path / fig_name
        fig.savefig(save_fig_path,dpi=300,format="png",bbox_inches="tight")

        #-----------------------------------------------------------------------
        # FIG 2: d outer limits with only sim shown
        fig,axs=plt.subplots(1,1,
                         figsize=plot_opts.single_fig_size_landscape,
                         layout="constrained")
        fig.set_dpi(plot_opts.resolution)

        # axs.ecdf(sim_cdfs_lims[kk]["nom"].quantiles,
        #          ls="-",color=sim_c,label="sim. nom.",linewidth=plot_opts.lw)
        axs.ecdf(sim_cdfs_lims[kk]["max"].quantiles,
                 ls=":",color=sim_c,label="sim. lims.",linewidth=plot_opts.lw)
        axs.ecdf(sim_cdfs_lims[kk]["min"].quantiles,
                 ls=":",color=sim_c,linewidth=plot_opts.lw)

        axs.fill_betweenx(sim_cdfs_lims[kk]["nom"].probabilities,
                         sim_cdfs_lims[kk]["min"].quantiles,
                         sim_cdfs_lims[kk]["max"].quantiles,
                         color=sim_c,
                         alpha=0.2,
                         ls=":")

        dplus_plot = dplus_max[kk]
        dminus_plot = dminus_max[kk]
        axs.fill_betweenx(dplus_plot["F_Y"],
                          dminus_plot["F_"] - dminus_plot["d-"],
                          dplus_plot["F_"] + dplus_plot["d+"],
                         color="black",
                         alpha=0.2,
                         ls=":")

        axs.ecdf(sim_cdfs_lims[kk]["max"].quantiles,
                 ls="-",color=sim_c,label="sim. lims.",linewidth=plot_opts.lw)
        axs.ecdf(sim_cdfs_lims[kk]["min"].quantiles,
                 ls="-",color=sim_c,linewidth=plot_opts.lw)

        axs.fill_betweenx(sim_cdfs_lims[kk]["nom"].probabilities,
                         sim_cdfs_lims[kk]["min"].quantiles,
                         sim_cdfs_lims[kk]["max"].quantiles,
                         color=sim_c,
                         alpha=0.2,
                         ls=":")

        dm_c = "black"
        axs.plot(dplus_plot["F_"] + dplus_plot["d+"],
                 dplus_plot["F_Y"],
                 ls=":",linewidth=plot_opts.lw*1.2, label="d+",
                 color=dm_c)
        axs.plot(dminus_plot["F_"] - dminus_plot["d-"],
                 dminus_plot["F_Y"],
                 ls="--",linewidth=plot_opts.lw*1.2, label="d-",
                 color=dm_c)


        title_str = f"{kk}"
        #axs.legend(loc="upper left",fontsize=6)
        axs.set_title(title_str,fontsize=plot_opts.font_head_size)
        axs.set_xlabel(sens_ax_labels[ii],fontsize=plot_opts.font_ax_size)
        axs.set_ylabel("Probability",fontsize=plot_opts.font_ax_size)

        fig_name = f"{fig_ind+1}_dextremes_simcdfsonly_{kk}.png"
        ax_lims[fig_name] = axs.get_xlim()
        save_fig_path = save_path / fig_name
        fig.savefig(save_fig_path,dpi=300,format="png",bbox_inches="tight")


    save_axlim_path = save_path / "axis_limits.json"
    with open(save_axlim_path, "w") as file:
        json.dump(ax_lims, file, indent=4)    


    #---------------------------------------------------------------------------
    # Save MAVM to csv for all cases
    print(80*"-")
    print("Saving MAVM results to csv.")

    n_res = len(mavm)
    d_res = np.zeros((n_res,2*4))
    d_rows = []
    d_cols = []

    print(f"{n_res=}")

    for ii,mm in enumerate(mavm): # Loop over sensors: TCs + CV
        d_rows.append(mm)

        for jj,es in enumerate(mavm[mm]):
            print(80*"-")
            print(f"{ii=}")
            print(f"{mm=}")
            print(f"{jj=}")
            print(f"{es=}")
            print(80*"-")

            if len(d_cols) != 8:
                d_cols.append(f"{es}-d+")
                d_cols.append(f"{es}-d-")

            d_res[ii,2*jj] = mavm[mm][es]["d+"]
            d_res[ii,2*jj+1] = mavm[mm][es]["d-"]

    data_frame = pd.DataFrame(d_res.T, columns=d_rows, index=d_cols)
    print(80*"-")
    print("MAVM Table")
    print(data_frame)

    save_mavm = save_path / "pointsensors_mavm.csv"
    data_frame.to_csv(save_mavm, index=True, header=True)

    #plt.show()

    print(80*"-")
    print("MAVM calculation complete.")
    print(80*"-")

    #---------------------------------------------------------------------------
    # plt.show()


    #---------------------------------------------------------------------------
    # Save d+ max and d- max values to csv
    
    d_extremes = []
    
    for kk in dplus_max.keys():
        d_extremes.append({
            "Sensor": kk,
            "d+_max": dplus_max[kk]["d+"],
            "d-_max": dminus_max[kk]["d-"],
            "d+_stat_key": dplus_max[kk]["stat_key"],
            "d-_stat_key": dminus_max[kk]["stat_key"],
        })
    
    d_extremes_df = pd.DataFrame(d_extremes)
    
    print(80*"-")
    print("Extreme d values")
    print(d_extremes_df)
    
    save_dextremes = save_path / "pointsensors_dextremes_all.csv"
    d_extremes_df.to_csv(save_dextremes, index=False)

    #---------------------------------------------------------------------------
    # Save d+ max and d- max values to csv
    
    # d_extremes = []
    
    # for kk in dplus_max.keys():
    #     d_extremes.append({
    #         "Sensor": kk,
    #         "d+_max": dplus_max[kk]["d+"],
    #         "d-_max": dminus_max[kk]["d-"],
    #     })
    
    # d_extremes_df = pd.DataFrame(d_extremes).T
    
    # print(80*"-")
    # print("Extreme d values")
    # print(d_extremes_df)
    
    # save_dextremes = save_path / "pointsensors_dextremes.csv"
    # d_extremes_df.to_csv(save_dextremes, index=False)


    d_extremes_df = pd.DataFrame(
    [
        [dplus_max[kk]["d+"] for kk in dplus_max],
        [dminus_max[kk]["d-"] for kk in dminus_max],
    ],
    index=["d+_max", "d-_max"],
    columns=list(dplus_max.keys()),
    )
    
    print(d_extremes_df)
    
    save_dextremes = save_path / "pointsensors_dextremes.csv"
    d_extremes_df.to_csv(save_dextremes, index=True, header=True)



    #---------------------------------------------------------------------------
    # Calculate total uncertainty for interpolation
    #
    # Total uncertainty =
    #     simulation epistemic uncertainty
    #     +
    #     MAVM-derived model-form uncertainty
    #
    # The MAVM d+ and d- values are horisontal CDF shifts, so we combine them
    # with the simulation CDF limits in the same output units.
    # We calculate a scalar upper/lower uncertainty over a probability range.
    #---------------------------------------------------------------------------

    print(80*"-")
    print("Calculating total uncertainty...")

    # Probability range over which the uncertainty is evaluated
    p_grid = np.linspace(0.05, 0.95, 181)

    total_uncertainty = {}
    total_bounds = {}

    def ecdf_quantile_at_probability(ecdf_obj, p_values):
        """
        Linearly interpolate ECDF quantiles at specified probabilities.
        """
        return np.interp(
            p_values,
            ecdf_obj.probabilities,
            ecdf_obj.quantiles,
        )

    for kk in dplus_max:

        sim_nom_cdf = sim_cdfs_lims[kk]["nom"]
        sim_max_cdf = sim_cdfs_lims[kk]["max"]
        sim_min_cdf = sim_cdfs_lims[kk]["min"]

        q_sim_nom = ecdf_quantile_at_probability(
            sim_nom_cdf,
            p_grid,
        )

        q_sim_max = ecdf_quantile_at_probability(
            sim_max_cdf,
            p_grid,
        )

        q_sim_min = ecdf_quantile_at_probability(
            sim_min_cdf,
            p_grid,
        )

        # Simulation epistemic uncertainty
        sim_unc_plus = q_sim_max - q_sim_nom
        sim_unc_minus = q_sim_nom - q_sim_min

        # MAVM/model-form contribution
        d_plus = dplus_max[kk]["d+"]
        d_minus = dminus_max[kk]["d-"]

        # Total uncertainty
        total_unc_plus_curve = sim_unc_plus + d_plus
        total_unc_minus_curve = sim_unc_minus + d_minus

        # Conservative scalar value for interpolation
        total_unc_plus = np.max(total_unc_plus_curve)
        total_unc_minus = np.max(total_unc_minus_curve)

        # Simulation-only components
        sim_unc_plus_max = np.max(sim_unc_plus)
        sim_unc_minus_max = np.max(sim_unc_minus)

        total_uncertainty[kk] = {
            "sim_plus": sim_unc_plus_max,
            "sim_minus": sim_unc_minus_max,
            "model_form_plus": d_plus,
            "model_form_minus": d_minus,
            "total_plus": total_unc_plus,
            "total_minus": total_unc_minus,
        }

        # Probability-dependent curves too
        total_bounds[kk] = {
            "p": p_grid,
            "q_sim_nom": q_sim_nom,
            "q_sim_min": q_sim_min,
            "q_sim_max": q_sim_max,

            "sim_unc_plus": sim_unc_plus,
            "sim_unc_minus": sim_unc_minus,

            "model_form_plus": np.full_like(
                p_grid,
                d_plus,
            ),
            "model_form_minus": np.full_like(
                p_grid,
                d_minus,
            ),

            "total_unc_plus": total_unc_plus_curve,
            "total_unc_minus": total_unc_minus_curve,

            # Final total bounds relative to nominal simulation.
            "q_total_upper": q_sim_nom + total_unc_plus_curve,
            "q_total_lower": q_sim_nom - total_unc_minus_curve,
        }

        print(80*"-")
        print(f"{kk=}")
        print(f"simulation + = {sim_unc_plus_max}")
        print(f"simulation - = {sim_unc_minus_max}")
        print(f"model form + = {d_plus}")
        print(f"model form - = {d_minus}")
        print(f"TOTAL + = {total_unc_plus}")
        print(f"TOTAL - = {total_unc_minus}")


    # Scalar total uncertainty for interpolation
    total_unc_df = pd.DataFrame(
        {
            kk: {
                "sim_plus": total_uncertainty[kk]["sim_plus"],
                "sim_minus": total_uncertainty[kk]["sim_minus"],
                "model_form_plus": total_uncertainty[kk]["model_form_plus"],
                "model_form_minus": total_uncertainty[kk]["model_form_minus"],
                "total_plus": total_uncertainty[kk]["total_plus"],
                "total_minus": total_uncertainty[kk]["total_minus"],
            }
            for kk in total_uncertainty
        }
    )

    print(80*"-")
    print("Total uncertainty table")
    print(total_unc_df)

    save_total_unc = save_path / "pointsensors_total_uncertainty.csv"
    total_unc_df.to_csv(
        save_total_unc,
        index=True,
        header=True,
    )

    temp_sensors = [
        kk for kk in total_unc_df.columns
        if kk != "CV"
    ]

    total_unc_temp_df = total_unc_df[temp_sensors]

    print(80*"-")
    print("Temperature sensor total uncertainty table")
    print(total_unc_temp_df)

    save_total_unc_temp = (
        save_path / "pointsensors_total_uncertainty_temperature.csv"
    )

    total_unc_temp_df.to_csv(
        save_total_unc_temp,
        index=True,
        header=True,
    )



    #---------------------------------------------------------------------------
    # Calculate mean simulation temperature
    #---------------------------------------------------------------------------

    sim_temp_df = pd.DataFrame(
    {
        kk: np.mean(sim_data_lims[kk]["nom"])
        for kk in sim_data_lims
        if kk != "CV"
    },
    index=["sim_nom_mean"]
    )

    save_mean_temp = (
        save_path / "sim_nom_mean_temperature.csv"
    )

    sim_temp_df.to_csv(
        save_mean_temp,
        index=True,
        header=True,
    )

    print(80*"-")
    print("Mean nominal simulation tamperature table")
    print(sim_temp_df)



if __name__ == "__main__":
    main()
