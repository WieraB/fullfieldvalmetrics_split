from pathlib import Path
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
    #SIM_DIR = Path.cwd() / "STC_ProbSim_FieldsFull_25X"
    SIM_DIR = Path.cwd() / "STC_ProbSim_FieldsReduced_25X"
    
    sens_ax_labels = (r"Temp. [$^{\circ}C$]",)*10 + (r"Coil RMS Voltage [$V$]",)
    sens_tags = ("Temp",)*10 + ("Volts",)
    sens_num = len(sens_tags)

    save_path = Path.cwd() / "topyvale_mavm_pulse25X"
    if not save_path.is_dir():
        save_path.mkdir(exist_ok=True,parents=True)

    plot_opts = pyvale.sensorsim.PlotOptsGeneral()

    fig_ind: int = 0
    exp_c: str = "tab:orange"
    sim_c: str = "tab:blue"

    #---------------------------------------------------------------------------
    # Simulation: Load Data
    print(f"Loading simulation data from:\n    {SIM_DIR}")

    # Reduced: 5000 = 100 aleatory x 50 epistemic
    # Full: 400 aleatory x 250 epistemic
    # exp_data = exp_data.reshape(samps_n,epis_n,alea_n)
    #samps_n: int = 5000
    epis_n: int = 50
    alea_n: int = 100

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
                "CV":15,}

    sim_path = SIM_DIR / "SamplingResults.csv"
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

    for kk in sim_data: # For each sensor
        cdfs = []

        max_data = sim_data[kk][0,:]
        min_data = sim_data[kk][0,:]

        min_cdf = stats.ecdf(sim_data[kk][0,:]).cdf
        max_cdf = stats.ecdf(sim_data[kk][0,:]).cdf

        sum_min_cdf = np.sum(min_cdf.quantiles)
        sum_max_cdf = np.sum(max_cdf.quantiles)

        accum_cdf = np.zeros_like(max_cdf.quantiles)

        for ee in range(epis_n): # For each epistemic sample
            this_cdf = stats.ecdf(sim_data[kk][ee,:]).cdf
            this_cdf_sum = np.sum(this_cdf.quantiles)

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


    PLOT_ALL_SIM_CDFS = False

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

            save_fig_path = save_path / f"{fig_ind}_sim_allcdfs_{kk}.png"
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

        dplus_max_val = 0
        dminus_max_val = 0
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

                    if this_mavm[stat_key]["d+"] > dplus_max_val:
                        dplus_max_val = this_mavm[stat_key]["d+"]
                        dplus_max[kk] = this_mavm[stat_key]
                        dplus_max[kk]["stat_key"] = stat_key

                    if this_mavm[stat_key]["d-"] > dminus_max_val:
                        dminus_max_val = this_mavm[stat_key]["d-"]
                        dminus_max[kk] = this_mavm[stat_key]
                        dminus_max[kk]["stat_key"] = stat_key

                    # if PLOT_MAVM_INDIVIDUAL:
                    #     vm.mavm_figs(this_mavm[stat_key],
                    #                 title_str=kk,
                    #                 field_label=sens_ax_labels[ii],
                    #                 field_tag=sens_tags[ii],
                    #                 save_tag=stat_key,
                    #                 save_path=save_path)

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

        axs.legend(loc="upper left",fontsize=6)
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

        # print(80*"-")
        # print(f"{kk=}")
        # print(f"{dplus_max[kk]['d+']=}")
        # print(f"{dminus_max[kk]['d-']=}")
        # print(80*"-")

        axs.legend(loc="upper left",fontsize=6)
        axs.set_title(kk,fontsize=plot_opts.font_head_size)
        axs.set_xlabel(sens_ax_labels[ii],fontsize=plot_opts.font_ax_size)
        axs.set_ylabel("Probability",fontsize=plot_opts.font_ax_size)

        save_fig_path = save_path / f"{fig_ind}_dextremes_wcdfs_{kk}.png"
        fig.savefig(save_fig_path,dpi=300,format="png",bbox_inches="tight")


    plt.close("all")
    #---------------------------------------------------------------------------
    # FIGURES: cleaner d+/- extremes
    fig_ind += 1
    for ii,kk in enumerate(exp_data_lims): # Loop over sensors: TCs + CV

        if np.any(np.isnan(exp_data_lims[kk]["nom"])):
            continue

        #-----------------------------------------------------------------------
        # FIG 1: d+ max
        fig,axs=plt.subplots(1,1,
                         figsize=plot_opts.single_fig_size_landscape,
                         layout="constrained")
        fig.set_dpi(plot_opts.resolution)

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
        dp_c = "black"
        axs.plot(dplus_max[kk]["F_"] + dplus_max[kk]["d+"],
                 dplus_max[kk]["F_Y"],
                 ls=":",linewidth=plot_opts.lw*1.2, label="d+",
                 color=dp_c)
        axs.plot(dplus_max[kk]["F_"] - dplus_max[kk]["d-"],
                 dplus_max[kk]["F_Y"],
                 ls="--",linewidth=plot_opts.lw*1.2, label="d-",
                 color=dp_c)


        title_str = f"{kk}, MAVM d+ maximised"
        axs.legend(loc="upper left",fontsize=6)
        axs.set_title(title_str,fontsize=plot_opts.font_head_size)
        axs.set_xlabel(sens_ax_labels[ii],fontsize=plot_opts.font_ax_size)
        axs.set_ylabel("Probability",fontsize=plot_opts.font_ax_size)

        save_fig_path = save_path / f"{fig_ind}_mavm_dplusmax_{kk}.png"
        fig.savefig(save_fig_path,dpi=300,format="png",bbox_inches="tight")

        plt.close(fig)

        #-----------------------------------------------------------------------
        # FIG 2: d- max
        fig,axs=plt.subplots(1,1,
                         figsize=plot_opts.single_fig_size_landscape,
                         layout="constrained")
        fig.set_dpi(plot_opts.resolution)

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
        axs.plot(dminus_max[kk]["F_"] + dminus_max[kk]["d+"],
                 dminus_max[kk]["F_Y"],
                 ls=":",linewidth=plot_opts.lw*1.2, label="d+",
                 color=dm_c)
        axs.plot(dminus_max[kk]["F_"] - dminus_max[kk]["d-"],
                 dminus_max[kk]["F_Y"],
                 ls="--",linewidth=plot_opts.lw*1.2, label="d-",
                 color=dm_c)


        title_str = f"{kk}, MAVM d- maximised"
        axs.legend(loc="upper left",fontsize=6)
        axs.set_title(title_str,fontsize=plot_opts.font_head_size)
        axs.set_xlabel(sens_ax_labels[ii],fontsize=plot_opts.font_ax_size)
        axs.set_ylabel("Probability",fontsize=plot_opts.font_ax_size)

        save_fig_path = save_path / f"{fig_ind+1}_mavm_dminusmax_{kk}.png"
        fig.savefig(save_fig_path,dpi=300,format="png",bbox_inches="tight")

        plt.close(fig)

        #-----------------------------------------------------------------------
        # FIG 3: d outer limits
        fig,axs=plt.subplots(1,1,
                         figsize=plot_opts.single_fig_size_landscape,
                         layout="constrained")
        fig.set_dpi(plot_opts.resolution)

        if ((np.mean(dplus_max[kk]["F_"]) + dplus_max[kk]["d+"])
            > (np.mean(dminus_max[kk]["F_"]) + dminus_max[kk]["d+"])):
            dplus_plot = dplus_max[kk]
        else:
            dplus_plot = dminus_max[kk]

        if ((np.mean(dplus_max[kk]["F_"]) - dplus_max[kk]["d-"])
             < (np.mean(dminus_max[kk]["F_"]) - dminus_max[kk]["d-"])):
            dminus_plot = dplus_max[kk]
        else:
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


        title_str = f"{kk}, MAVM limits"
        axs.legend(loc="upper left",fontsize=6)
        axs.set_title(title_str,fontsize=plot_opts.font_head_size)
        axs.set_xlabel(sens_ax_labels[ii],fontsize=plot_opts.font_ax_size)
        axs.set_ylabel("Probability",fontsize=plot_opts.font_ax_size)

        save_fig_path = save_path / f"{fig_ind+2}_dextremes_wcdfs_{kk}.png"
        fig.savefig(save_fig_path,dpi=300,format="png",bbox_inches="tight")


    plt.show()

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


if __name__ == "__main__":
    main()
