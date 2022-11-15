import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.legend_handler import HandlerErrorbar

"""
Goal of program is to read .csv files containing results from experiments and
make plots / analyse results.
"""

def generate_errorbar(results_file, doses_file):
    # read files
    df = pd.read_csv(results_file)
    df = df.drop(columns=["irradiation date","cell line","radiation type","image_nr", "sample_id"])
    df = df.astype(np.float_)
    doses = pd.read_csv(doses_file).astype(np.float_)

    # insert 0 for the control sample position
    df.position[pd.isna(df.position)] = 0

    # make columns for prescribed dose, mean dose and std
    df["prescribed"] = 0
    df["meandose"] = 0
    df["stddose"] = 0

    for pos in doses.position.unique():
        for dose in doses.givendose.unique():
            # collect values in doses
            locs = (doses.position == pos) & (doses.givendose == dose)
            if np.any(locs):
                prescribed = doses[locs].prescribed.unique()[0]
                meandose = doses[locs].meandose.unique()[0]
                stddose = doses[locs].stddose.unique()[0]

                # fill values in df
                locs = (df.position == pos) & (df.dose == dose)
                df.loc[locs,"prescribed"] = prescribed
                df.loc[locs,"meandose"] = meandose
                df.loc[locs,"stddose"] = stddose

    # find all binucleated cells
    df_bi = df[df.nuclei == 2].drop(columns=["nuclei","prescribed"])

    # compute statistics
    DF = df_bi.groupby(["dose","position"]).agg(
        mean_MN = ("micronuclei", "mean"),
        sdem_MN = ("micronuclei", "sem"),
        meandose = ("meandose", "max"),  # mean returns same value
        stddose = ("stddose", "max")     # mean returns same value
    ).reset_index()

    DF = DF.groupby(["position", "meandose"]).agg(
        mean_MN = ("mean_MN", "mean"),
        sdem_MN = ("sdem_MN", lambda x: np.sqrt(np.mean(x**2))),
        stddose = ("stddose", "max")     # mean returns same value
    ).reset_index()

    # split dataframe
    DF0 = DF[DF.position == 0].drop(columns=["position"])
    DF1 = DF[DF.position == 1].drop(columns=["position"])
    DF5 = DF[DF.position == 5].drop(columns=["position"])

    return DF0, DF1, DF5

# -- PROTON EXPERIMENTS --

# 15MAR2022
DF0_15MAR, DF1_15MAR, DF5_15MAR = generate_errorbar("./final_results/15MAR2022_results.csv", "15MAR2022_doses.csv")

# 16MAR2022
DF0_16MAR, DF1_16MAR, DF5_16MAR = generate_errorbar("./final_results/16MAR2022_results.csv", "16MAR2022_doses.csv")

# plot side by side
figure, axes = plt.subplots(1,2,sharey=True)

axes[0].set_title("15MAR2022", fontsize=15)
axes[0].errorbar(DF1_15MAR.meandose, DF1_15MAR.mean_MN,
                 xerr=DF1_15MAR.stddose, yerr=DF1_15MAR.sdem_MN,
                 color="darkblue", label="p1", ls="none", capsize=4, marker="D")
axes[0].errorbar(DF5_15MAR.meandose, DF5_15MAR.mean_MN,
                 xerr=DF5_15MAR.stddose, yerr=DF5_15MAR.sdem_MN,
                 color="darkviolet", label="p5", ls="none", capsize=4, marker="D")
xlim = axes[0].get_xlim()
axes[0].fill_between(xlim, [(DF0_15MAR.mean_MN[0]+DF0_15MAR.sdem_MN[0]),
                     (DF0_15MAR.mean_MN[0]+DF0_15MAR.sdem_MN[0])],
                     [(DF0_15MAR.mean_MN[0]-DF0_15MAR.sdem_MN[0]),
                     (DF0_15MAR.mean_MN[0]-DF0_15MAR.sdem_MN[0])], color="skyblue", alpha=0.4)
axes[0].hlines(y=DF0_15MAR.mean_MN, xmin=xlim[0], xmax=xlim[1], label="Control", color="dodgerblue")
axes[0].legend()
axes[0].set_xlabel("Dose [Gy]", fontsize=15)
axes[0].set_ylabel("Average number of micronuclei\nin binucleated cells", fontsize=15)
axes[0].set_xlim(xlim)

axes[1].set_title("16MAR2022", fontsize=15)
axes[1].errorbar(DF1_16MAR.meandose, DF1_16MAR.mean_MN,
             xerr=DF1_16MAR.stddose, yerr=DF1_16MAR.sdem_MN,
             color="darkblue", label="p1", ls="none", capsize=4, marker="D")
axes[1].errorbar(DF5_16MAR.meandose, DF5_16MAR.mean_MN,
                 xerr=DF5_16MAR.stddose, yerr=DF5_16MAR.sdem_MN,
                 color="darkviolet", label="p5", ls="none", capsize=4, marker="D")
xlim = axes[1].get_xlim()
axes[1].fill_between(xlim, [(DF0_16MAR.mean_MN[0]+DF0_16MAR.sdem_MN[0]),
                     (DF0_16MAR.mean_MN[0]+DF0_16MAR.sdem_MN[0])],
                     [(DF0_16MAR.mean_MN[0]-DF0_16MAR.sdem_MN[0]),
                     (DF0_16MAR.mean_MN[0]-DF0_16MAR.sdem_MN[0])], alpha=0.4, color="skyblue")
axes[1].hlines(y=DF0_16MAR.mean_MN, xmin=xlim[0], xmax=xlim[1], label="Control", color="dodgerblue")
axes[1].legend()
axes[1].set_xlabel("Dose [Gy]", fontsize=15)
axes[1].set_xlim(xlim)

figure.suptitle("Micronucleus production in A549-cells\nfollowing proton irradiation", fontsize=20)

figure.tight_layout()

plt.savefig("./Figures/MN_A549_protons.png", bbox_inches="tight", dpi=200)
plt.show()


# 05MAY2022
DF0_05MAY, DF1_05MAY, DF5_05MAY = generate_errorbar("./final_results/05MAY2022_results.csv", "05MAY2022_doses.csv")

# 06SEP2022
DF0_06SEP, DF1_06SEP, DF5_06SEP = generate_errorbar("./final_results/06SEP2022_results.csv", "06SEP2022_doses.csv")

# plot side by side
figure, axes = plt.subplots(1,2,sharey=True)

axes[0].set_title("05MAY2022", fontsize=15)
axes[0].errorbar(DF1_05MAY.meandose, DF1_05MAY.mean_MN,
                 xerr=DF1_05MAY.stddose, yerr=DF1_05MAY.sdem_MN,
                 color="darkblue", label="p1", ls="none", capsize=4, marker="D")
axes[0].errorbar(DF5_05MAY.meandose, DF5_05MAY.mean_MN,
                 xerr=DF5_05MAY.stddose, yerr=DF5_05MAY.sdem_MN,
                 color="darkviolet", label="p5", ls="none", capsize=4, marker="D")
xlim = axes[0].get_xlim()
axes[0].fill_between(xlim, [(DF0_05MAY.mean_MN[0]+DF0_05MAY.sdem_MN[0]),
                     (DF0_05MAY.mean_MN[0]+DF0_05MAY.sdem_MN[0])],
                     [(DF0_05MAY.mean_MN[0]-DF0_05MAY.sdem_MN[0]),
                     (DF0_05MAY.mean_MN[0]-DF0_05MAY.sdem_MN[0])], color="skyblue", alpha=0.4)
axes[0].hlines(y=DF0_05MAY.mean_MN, xmin=xlim[0], xmax=xlim[1], label="Control", color="dodgerblue")
axes[0].legend()
axes[0].set_xlabel("Dose [Gy]", fontsize=15)
axes[0].set_ylabel("Average number of micronuclei\nin binucleated cells", fontsize=15)
axes[0].set_xlim(xlim)

axes[1].set_title("06SEP2022", fontsize=15)
axes[1].errorbar(DF1_06SEP.meandose, DF1_06SEP.mean_MN,
             xerr=DF1_06SEP.stddose, yerr=DF1_06SEP.sdem_MN,
             color="darkblue", label="p1", ls="none", capsize=4, marker="D")
axes[1].errorbar(DF5_06SEP.meandose, DF5_06SEP.mean_MN,
                 xerr=DF5_06SEP.stddose, yerr=DF5_06SEP.sdem_MN,
                 color="darkviolet", label="p5", ls="none", capsize=4, marker="D")
xlim = axes[1].get_xlim()
axes[1].fill_between(xlim, [(DF0_06SEP.mean_MN[0]+DF0_06SEP.sdem_MN[0]),
                     (DF0_06SEP.mean_MN[0]+DF0_06SEP.sdem_MN[0])],
                     [(DF0_06SEP.mean_MN[0]-DF0_06SEP.sdem_MN[0]),
                     (DF0_06SEP.mean_MN[0]-DF0_06SEP.sdem_MN[0])], alpha=0.4, color="skyblue")
axes[1].hlines(y=DF0_06SEP.mean_MN, xmin=xlim[0], xmax=xlim[1], label="Control", color="dodgerblue")
axes[1].legend()
axes[1].set_xlabel("Dose [Gy]", fontsize=15)
axes[1].set_xlim(xlim)

figure.suptitle("Micronucleus production in MOC2-cells\nfollowing proton irradiation", fontsize=20)

figure.tight_layout()
plt.savefig("./Figures/MN_MOC2_protons.png", bbox_inches="tight", dpi=200)
plt.show()


# -- X-RAYS --

# 03JUN2022
df_03JUN = pd.read_csv("./final_results/03JUN2022_results.csv")
df_03JUN = df_03JUN.drop(columns=["irradiation date","cell line","radiation type","image_nr"])

# find all binucleated cells
df_bi_03JUN = df_03JUN[df_03JUN.nuclei == 2].drop(columns=["nuclei"])

# compute mean and standard error of MN
DF_03JUN = df_bi_03JUN.groupby(["dose"]).agg(mean_MN = ("micronuclei", "mean"),
                                          sdem_MN = ("micronuclei", "sem")
                                          ).reset_index()

# 19SEP2022
df_19SEP = pd.read_csv("./final_results/19SEP2022_results.csv")
df_19SEP = df_19SEP.drop(columns=["irradiation date","cell line","radiation type","image_nr"])

# find all binucleated cells
df_bi_19SEP = df_19SEP[df_19SEP.nuclei == 2].drop(columns=["nuclei"])

# compute mean and standard error of MN
DF_19SEP = df_bi_19SEP.groupby(["dose"]).agg(mean_MN = ("micronuclei", "mean"),
                                          sdem_MN = ("micronuclei", "sem")
                                          ).reset_index()

# plot side-by-side
figure, axes = plt.subplots(1,2,sharey=False)

axes[0].errorbar(DF_03JUN.dose[1:], DF_03JUN.mean_MN[1:], yerr=DF_03JUN.sdem_MN[1:],
             color="darkblue", ls="none", capsize=4, marker="D")
xlim = axes[0].get_xlim()
axes[0].fill_between(xlim, [(DF_03JUN.mean_MN[0]+DF_03JUN.sdem_MN[0]),
                  (DF_03JUN.mean_MN[0]+DF_03JUN.sdem_MN[0])],
                  [(DF_03JUN.mean_MN[0]-DF_03JUN.sdem_MN[0]),
                  (DF_03JUN.mean_MN[0]-DF_03JUN.sdem_MN[0])], color="skyblue", alpha=0.4)
axes[0].hlines(y=DF_03JUN.mean_MN[0], xmin=xlim[0], xmax=xlim[1], label="Control", color="dodgerblue")
axes[0].set_xlabel("Dose [Gy]", fontsize=15)
axes[0].set_ylabel("Average number of micronuclei\nin binucleated cells", fontsize=15)
axes[0].set_xlim(xlim)
axes[0].legend()
axes[0].set_title("A549\n03JUN2022", fontsize=15)

axes[1].errorbar(DF_19SEP.dose[1:], DF_19SEP.mean_MN[1:], yerr=DF_19SEP.sdem_MN[1:],
             color="darkblue", ls="none", capsize=4, marker="D")
xlim = axes[1].get_xlim()
axes[1].fill_between(xlim, [(DF_19SEP.mean_MN[0]+DF_19SEP.sdem_MN[0]),
                  (DF_19SEP.mean_MN[0]+DF_19SEP.sdem_MN[0])],
                  [(DF_19SEP.mean_MN[0]-DF_19SEP.sdem_MN[0]),
                  (DF_19SEP.mean_MN[0]-DF_19SEP.sdem_MN[0])], color="skyblue", alpha=0.4)
axes[1].hlines(y=DF_19SEP.mean_MN[0], xmin=xlim[0], xmax=xlim[1], label="Control", color="dodgerblue")
axes[1].set_xlabel("Dose [Gy]", fontsize=15)
axes[1].set_xlim(xlim)
axes[1].legend()
axes[1].set_title("MOC2\n19SEP2022", fontsize=15)

figure.suptitle("Micornucleus production following\nx-ray irradiation", fontsize=20)
figure.tight_layout()

plt.savefig("./Figures/MN_A549_MOC2_xrays.png", bbox_inches="tight", dpi=200)
plt.show()


# collect all A549 experiments

# plot side by side
figure, axes = plt.subplots(1,3,sharey=True, figsize=(10,5))

axes[0].set_title("Protons\n15MAR2022", fontsize=15)
axes[0].errorbar(DF1_15MAR.meandose, DF1_15MAR.mean_MN,
                 xerr=DF1_15MAR.stddose, yerr=DF1_15MAR.sdem_MN,
                 color="darkblue", label="p1", ls="none", capsize=4, marker="D")
axes[0].errorbar(DF5_15MAR.meandose, DF5_15MAR.mean_MN,
                 xerr=DF5_15MAR.stddose, yerr=DF5_15MAR.sdem_MN,
                 color="darkviolet", label="p5", ls="none", capsize=4, marker="D")
xlim = axes[0].get_xlim()
axes[0].fill_between(xlim, [(DF0_15MAR.mean_MN[0]+DF0_15MAR.sdem_MN[0]),
                     (DF0_15MAR.mean_MN[0]+DF0_15MAR.sdem_MN[0])],
                     [(DF0_15MAR.mean_MN[0]-DF0_15MAR.sdem_MN[0]),
                     (DF0_15MAR.mean_MN[0]-DF0_15MAR.sdem_MN[0])], color="skyblue", alpha=0.4)
axes[0].hlines(y=DF0_15MAR.mean_MN, xmin=xlim[0], xmax=xlim[1], label="Control", color="dodgerblue")
axes[0].legend()
axes[0].set_xlabel("Dose [Gy]", fontsize=15)
axes[0].set_ylabel("Average number of micronuclei\nin binucleated cells", fontsize=15)
axes[0].set_xlim(xlim)

axes[1].set_title("Protons\n16MAR2022", fontsize=15)
axes[1].errorbar(DF1_16MAR.meandose, DF1_16MAR.mean_MN,
             xerr=DF1_16MAR.stddose, yerr=DF1_16MAR.sdem_MN,
             color="darkblue", label="p1", ls="none", capsize=4, marker="D")
axes[1].errorbar(DF5_16MAR.meandose, DF5_16MAR.mean_MN,
                 xerr=DF5_16MAR.stddose, yerr=DF5_16MAR.sdem_MN,
                 color="darkviolet", label="p5", ls="none", capsize=4, marker="D")
xlim = axes[1].get_xlim()
axes[1].fill_between(xlim, [(DF0_16MAR.mean_MN[0]+DF0_16MAR.sdem_MN[0]),
                     (DF0_16MAR.mean_MN[0]+DF0_16MAR.sdem_MN[0])],
                     [(DF0_16MAR.mean_MN[0]-DF0_16MAR.sdem_MN[0]),
                     (DF0_16MAR.mean_MN[0]-DF0_16MAR.sdem_MN[0])], alpha=0.4, color="skyblue")
axes[1].hlines(y=DF0_16MAR.mean_MN, xmin=xlim[0], xmax=xlim[1], label="Control", color="dodgerblue")
axes[1].legend()
axes[1].set_xlabel("Dose [Gy]", fontsize=15)
axes[1].set_xlim(xlim)

axes[2].errorbar(DF_03JUN.dose[1:], DF_03JUN.mean_MN[1:], yerr=DF_03JUN.sdem_MN[1:],
             color="darkblue", ls="none", capsize=4, marker="D", label="Irradiated samples")
xlim = axes[2].get_xlim()
axes[2].fill_between(xlim, [(DF_03JUN.mean_MN[0]+DF_03JUN.sdem_MN[0]),
                  (DF_03JUN.mean_MN[0]+DF_03JUN.sdem_MN[0])],
                  [(DF_03JUN.mean_MN[0]-DF_03JUN.sdem_MN[0]),
                  (DF_03JUN.mean_MN[0]-DF_03JUN.sdem_MN[0])], color="skyblue", alpha=0.4)
axes[2].hlines(y=DF_03JUN.mean_MN[0], xmin=xlim[0], xmax=xlim[1], label="Control", color="dodgerblue")
axes[2].set_xlabel("Dose [Gy]", fontsize=15)
axes[2].set_xlim(xlim)
axes[2].legend()
axes[2].set_title("X-rays\n03JUN2022", fontsize=15)

figure.suptitle("Micronucleus production in A549-cells", fontsize=20)

figure.tight_layout()

plt.savefig("./Figures/MN_A549.png", bbox_inches="tight", dpi=200)
plt.show()


# plot beneath one another
figure, axes = plt.subplots(3,1 ,sharey=True, figsize=(7,12))

axes[0].set_title("Protons\n15MAR2022", fontsize=15)
axes[0].errorbar(DF1_15MAR.meandose, DF1_15MAR.mean_MN,
                 xerr=DF1_15MAR.stddose, yerr=DF1_15MAR.sdem_MN,
                 color="darkblue", label="p1", ls="none", capsize=4, marker="D")
axes[0].errorbar(DF5_15MAR.meandose, DF5_15MAR.mean_MN,
                 xerr=DF5_15MAR.stddose, yerr=DF5_15MAR.sdem_MN,
                 color="darkviolet", label="p5", ls="none", capsize=4, marker="D")
axes[0].set_xlim(3,10.5)
xlim = axes[0].get_xlim()
axes[0].fill_between(xlim, [(DF0_15MAR.mean_MN[0]+DF0_15MAR.sdem_MN[0]),
                     (DF0_15MAR.mean_MN[0]+DF0_15MAR.sdem_MN[0])],
                     [(DF0_15MAR.mean_MN[0]-DF0_15MAR.sdem_MN[0]),
                     (DF0_15MAR.mean_MN[0]-DF0_15MAR.sdem_MN[0])], color="skyblue", alpha=0.4)
axes[0].hlines(y=DF0_15MAR.mean_MN, xmin=xlim[0], xmax=xlim[1], label="Control", color="dodgerblue")
axes[0].legend()
axes[0].set_ylabel("Average number of micronuclei\nin binucleated cells", fontsize=15)
axes[0].set_xlim(xlim)
axes[0].set_xlabel("Dose [Gy]", fontsize=15)

axes[1].set_title("Protons\n16MAR2022", fontsize=15)
axes[1].errorbar(DF1_16MAR.meandose, DF1_16MAR.mean_MN,
             xerr=DF1_16MAR.stddose, yerr=DF1_16MAR.sdem_MN,
             color="darkblue", label="p1", ls="none", capsize=4, marker="D")
axes[1].errorbar(DF5_16MAR.meandose, DF5_16MAR.mean_MN,
                 xerr=DF5_16MAR.stddose, yerr=DF5_16MAR.sdem_MN,
                 color="darkviolet", label="p5", ls="none", capsize=4, marker="D")
axes[1].set_xlim(3,10.5)
xlim = axes[1].get_xlim()
axes[1].fill_between(xlim, [(DF0_16MAR.mean_MN[0]+DF0_16MAR.sdem_MN[0]),
                     (DF0_16MAR.mean_MN[0]+DF0_16MAR.sdem_MN[0])],
                     [(DF0_16MAR.mean_MN[0]-DF0_16MAR.sdem_MN[0]),
                     (DF0_16MAR.mean_MN[0]-DF0_16MAR.sdem_MN[0])], alpha=0.4, color="skyblue")
axes[1].hlines(y=DF0_16MAR.mean_MN, xmin=xlim[0], xmax=xlim[1], label="Control", color="dodgerblue")
axes[1].legend()
axes[1].set_ylabel("Average number of micronuclei\nin binucleated cells", fontsize=15)
axes[1].set_xlim(xlim)
axes[1].set_xlabel("Dose [Gy]", fontsize=15)


axes[2].errorbar(DF_03JUN.dose[1:], DF_03JUN.mean_MN[1:], yerr=DF_03JUN.sdem_MN[1:],
             color="darkblue", ls="none", capsize=4, marker="D", label="Irradiated samples")
axes[2].set_xlim(3,10.5)
xlim = axes[2].get_xlim()
axes[2].fill_between(xlim, [(DF_03JUN.mean_MN[0]+DF_03JUN.sdem_MN[0]),
                  (DF_03JUN.mean_MN[0]+DF_03JUN.sdem_MN[0])],
                  [(DF_03JUN.mean_MN[0]-DF_03JUN.sdem_MN[0]),
                  (DF_03JUN.mean_MN[0]-DF_03JUN.sdem_MN[0])], color="skyblue", alpha=0.4)
axes[2].hlines(y=DF_03JUN.mean_MN[0], xmin=xlim[0], xmax=xlim[1], label="Control", color="dodgerblue")
axes[2].set_xlabel("Dose [Gy]", fontsize=15)
axes[2].set_xlim(xlim)
axes[2].legend()
axes[2].set_ylabel("Average number of micronuclei\nin binucleated cells", fontsize=15)
axes[2].set_title("X-rays\n03JUN2022", fontsize=15)

figure.suptitle("Micronucleus production in A549-cells", fontsize=20)

figure.tight_layout()
plt.savefig("./Figures/MN_A549_common_xaxes.png", bbox_inches="tight", dpi=200)
plt.show()


# collect all MOC2 experiments

# plot side by side
figure, axes = plt.subplots(1,3,sharey=True, figsize=(10,5))

axes[0].set_title("Protons\n05MAY2022", fontsize=15)
axes[0].errorbar(DF1_05MAY.meandose, DF1_05MAY.mean_MN,
                 xerr=DF1_05MAY.stddose, yerr=DF1_05MAY.sdem_MN,
                 color="darkblue", label="p1", ls="none", capsize=4, marker="D")
axes[0].errorbar(DF5_05MAY.meandose, DF5_05MAY.mean_MN,
                 xerr=DF5_05MAY.stddose, yerr=DF5_05MAY.sdem_MN,
                 color="darkviolet", label="p5", ls="none", capsize=4, marker="D")
xlim = axes[0].get_xlim()
axes[0].fill_between(xlim, [(DF0_05MAY.mean_MN[0]+DF0_05MAY.sdem_MN[0]),
                     (DF0_05MAY.mean_MN[0]+DF0_05MAY.sdem_MN[0])],
                     [(DF0_05MAY.mean_MN[0]-DF0_05MAY.sdem_MN[0]),
                     (DF0_05MAY.mean_MN[0]-DF0_05MAY.sdem_MN[0])], color="skyblue", alpha=0.4)
axes[0].hlines(y=DF0_05MAY.mean_MN, xmin=xlim[0], xmax=xlim[1], label="Control", color="dodgerblue")
axes[0].legend()
axes[0].set_xlabel("Dose [Gy]", fontsize=15)
axes[0].set_ylabel("Average number of micronuclei\nin binucleated cells", fontsize=15)
axes[0].set_xlim(xlim)

axes[1].set_title("Protons\n06SEP2022", fontsize=15)
axes[1].errorbar(DF1_06SEP.meandose, DF1_06SEP.mean_MN,
             xerr=DF1_06SEP.stddose, yerr=DF1_06SEP.sdem_MN,
             color="darkblue", label="p1", ls="none", capsize=4, marker="D")
axes[1].errorbar(DF5_06SEP.meandose, DF5_06SEP.mean_MN,
                 xerr=DF5_06SEP.stddose, yerr=DF5_06SEP.sdem_MN,
                 color="darkviolet", label="p5", ls="none", capsize=4, marker="D")
xlim = axes[1].get_xlim()
axes[1].fill_between(xlim, [(DF0_06SEP.mean_MN[0]+DF0_06SEP.sdem_MN[0]),
                     (DF0_06SEP.mean_MN[0]+DF0_06SEP.sdem_MN[0])],
                     [(DF0_06SEP.mean_MN[0]-DF0_06SEP.sdem_MN[0]),
                     (DF0_06SEP.mean_MN[0]-DF0_06SEP.sdem_MN[0])], alpha=0.4, color="skyblue")
axes[1].hlines(y=DF0_06SEP.mean_MN, xmin=xlim[0], xmax=xlim[1], label="Control", color="dodgerblue")
axes[1].legend()
axes[1].set_xlabel("Dose [Gy]", fontsize=15)
axes[1].set_xlim(xlim)

axes[2].errorbar(DF_19SEP.dose[1:], DF_19SEP.mean_MN[1:], yerr=DF_19SEP.sdem_MN[1:],
             color="darkblue", ls="none", capsize=4, marker="D", label="Irradiated samples")
xlim = axes[2].get_xlim()
axes[2].fill_between(xlim, [(DF_19SEP.mean_MN[0]+DF_19SEP.sdem_MN[0]),
                  (DF_19SEP.mean_MN[0]+DF_19SEP.sdem_MN[0])],
                  [(DF_19SEP.mean_MN[0]-DF_19SEP.sdem_MN[0]),
                  (DF_19SEP.mean_MN[0]-DF_19SEP.sdem_MN[0])], color="skyblue", alpha=0.4)
axes[2].hlines(y=DF_19SEP.mean_MN[0], xmin=xlim[0], xmax=xlim[1], label="Control", color="dodgerblue")
axes[2].set_xlabel("Dose [Gy]", fontsize=15)
axes[2].set_xlim(xlim)
axes[2].legend()
axes[2].set_title("X-rays\n19SEP2022", fontsize=15)

figure.suptitle("Micronucleus production in MOC2-cells", fontsize=20)
figure.tight_layout()

plt.savefig("./Figures/MN_MOC2.png", bbox_inches="tight", dpi=200)
plt.show()

# plot beneath one another
figure, axes = plt.subplots(3,1,sharey=True, figsize=(7,12))

axes[0].set_title("Protons\n05MAY2022", fontsize=15)
axes[0].errorbar(DF1_05MAY.meandose, DF1_05MAY.mean_MN,
                 xerr=DF1_05MAY.stddose, yerr=DF1_05MAY.sdem_MN,
                 color="darkblue", label="p1", ls="none", capsize=4, marker="D")
axes[0].errorbar(DF5_05MAY.meandose, DF5_05MAY.mean_MN,
                 xerr=DF5_05MAY.stddose, yerr=DF5_05MAY.sdem_MN,
                 color="darkviolet", label="p5", ls="none", capsize=4, marker="D")
axes[0].set_xlim(3,10.5)
xlim = axes[0].get_xlim()
axes[0].fill_between(xlim, [(DF0_05MAY.mean_MN[0]+DF0_05MAY.sdem_MN[0]),
                     (DF0_05MAY.mean_MN[0]+DF0_05MAY.sdem_MN[0])],
                     [(DF0_05MAY.mean_MN[0]-DF0_05MAY.sdem_MN[0]),
                     (DF0_05MAY.mean_MN[0]-DF0_05MAY.sdem_MN[0])], color="skyblue", alpha=0.4)
axes[0].hlines(y=DF0_05MAY.mean_MN, xmin=xlim[0], xmax=xlim[1], label="Control", color="dodgerblue")
axes[0].legend()
axes[0].set_xlabel("Dose [Gy]", fontsize=15)
axes[0].set_ylabel("Average number of micronuclei\nin binucleated cells", fontsize=15)
axes[0].set_xlim(xlim)

axes[1].set_title("Protons\n06SEP2022", fontsize=15)
axes[1].errorbar(DF1_06SEP.meandose, DF1_06SEP.mean_MN,
             xerr=DF1_06SEP.stddose, yerr=DF1_06SEP.sdem_MN,
             color="darkblue", label="p1", ls="none", capsize=4, marker="D")
axes[1].errorbar(DF5_06SEP.meandose, DF5_06SEP.mean_MN,
                 xerr=DF5_06SEP.stddose, yerr=DF5_06SEP.sdem_MN,
                 color="darkviolet", label="p5", ls="none", capsize=4, marker="D")
axes[1].set_xlim(3,10.5)
xlim = axes[1].get_xlim()
axes[1].fill_between(xlim, [(DF0_06SEP.mean_MN[0]+DF0_06SEP.sdem_MN[0]),
                     (DF0_06SEP.mean_MN[0]+DF0_06SEP.sdem_MN[0])],
                     [(DF0_06SEP.mean_MN[0]-DF0_06SEP.sdem_MN[0]),
                     (DF0_06SEP.mean_MN[0]-DF0_06SEP.sdem_MN[0])], alpha=0.4, color="skyblue")
axes[1].hlines(y=DF0_06SEP.mean_MN, xmin=xlim[0], xmax=xlim[1], label="Control", color="dodgerblue")
axes[1].legend()
axes[1].set_xlabel("Dose [Gy]", fontsize=15)
axes[1].set_ylabel("Average number of micronuclei\nin binucleated cells", fontsize=15)
axes[1].set_xlim(xlim)

axes[2].errorbar(DF_19SEP.dose[1:], DF_19SEP.mean_MN[1:], yerr=DF_19SEP.sdem_MN[1:],
             color="darkblue", ls="none", capsize=4, marker="D", label="Irradiated samples")
axes[2].set_xlim(3,10.5)
xlim = axes[2].get_xlim()
axes[2].fill_between(xlim, [(DF_19SEP.mean_MN[0]+DF_19SEP.sdem_MN[0]),
                  (DF_19SEP.mean_MN[0]+DF_19SEP.sdem_MN[0])],
                  [(DF_19SEP.mean_MN[0]-DF_19SEP.sdem_MN[0]),
                  (DF_19SEP.mean_MN[0]-DF_19SEP.sdem_MN[0])], color="skyblue", alpha=0.4)
axes[2].hlines(y=DF_19SEP.mean_MN[0], xmin=xlim[0], xmax=xlim[1], label="Control", color="dodgerblue")
axes[2].set_xlabel("Dose [Gy]", fontsize=15)
axes[2].set_ylabel("Average number of micronuclei\nin binucleated cells", fontsize=15)
axes[2].set_xlim(xlim)
axes[2].legend()
axes[2].set_title("X-rays\n19SEP2022", fontsize=15)

figure.suptitle("Micronucleus production in MOC2-cells", fontsize=20)
figure.tight_layout()

plt.savefig("./Figures/MN_MOC2_common_xaxes.png", bbox_inches="tight", dpi=200)
plt.show()


# comparison

plt.figure(figsize=(6.5,6))
plt.grid(color="lightgray", axis="y")
h1 = plt.errorbar(DF1_16MAR.meandose, DF1_16MAR.mean_MN.values/DF0_16MAR.mean_MN.values,
             xerr=DF1_16MAR.stddose, yerr=DF1_16MAR.sdem_MN.values/DF0_16MAR.mean_MN.values,
             color="blue", label="A549, protons, p1", ls="none", capsize=10, marker="s", ms=8)
h2 = plt.errorbar(DF5_16MAR.meandose, DF5_16MAR.mean_MN.values/DF0_16MAR.mean_MN.values,
                 xerr=DF5_16MAR.stddose, yerr=DF5_16MAR.sdem_MN.values/DF0_16MAR.mean_MN.values,
                 color="magenta", label="A549, protons, p5", ls="none", capsize=10, marker="s", ms=8)

h3 = plt.errorbar(DF1_05MAY.meandose, DF1_05MAY.mean_MN.values/DF0_05MAY.mean_MN.values,
             xerr=DF1_05MAY.stddose, yerr=DF1_05MAY.sdem_MN.values/DF0_05MAY.mean_MN.values,
             color="blue", label="MOC2, protons, p1", ls="none", capsize=10, marker="^", ms=8)
h4 = plt.errorbar(DF5_05MAY.meandose, DF5_05MAY.mean_MN.values/DF0_05MAY.mean_MN.values,
                 xerr=DF5_05MAY.stddose, yerr=DF5_05MAY.sdem_MN.values/DF0_05MAY.mean_MN.values,
                 color="magenta", label="MOC2, protons, p5", ls="none", capsize=10, marker="^", ms=8)

h5 = plt.errorbar(DF_03JUN.dose[1:], DF_03JUN.mean_MN[1:].values/DF_03JUN.mean_MN[0],
             yerr=DF_03JUN.sdem_MN[1:].values/DF_03JUN.sdem_MN[0],
             color="green", ls="none", capsize=15, marker="s", label="A549, x-rays", ms=8)
plt.ylim(0,6)

plt.legend(loc="upper left", fontsize=13, markerscale=0.8, handler_map={
    type(h1): HandlerErrorbar(xerr_size=0.65,yerr_size=0.65),
    type(h2): HandlerErrorbar(xerr_size=0.65,yerr_size=0.65),
    type(h3): HandlerErrorbar(xerr_size=0.65,yerr_size=0.65),
    type(h4): HandlerErrorbar(xerr_size=0.65,yerr_size=0.65),
    type(h5): HandlerErrorbar(xerr_size=0.65,yerr_size=0.65)})
plt.xlabel("Dose [Gy]", fontsize=15)
plt.ylabel("Micornucleus production scale factor", fontsize=15)
plt.title("Relative micronucleus production", fontsize=20)

plt.tight_layout()
plt.savefig("./Figures/results_summary.png", bbox_inches="tight", dpi=200)
plt.show()
