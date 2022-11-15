import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

xlim = [0,7]
x = [1,2,3,4,5,6]
vals_bn = []
vals_sn = []
vals_tot = []
vals_tn = []
vals_nn = []

df = pd.read_csv("./final_results/15MAR2022_results.csv")
df = df.drop(columns=["irradiation date","cell line","radiation type","image_nr"])
df = df.astype(np.float_)
vals_bn.append(len(df[df.nuclei==2]))
vals_tot.append(len(df))
vals_sn.append(len(df[df.nuclei==1]))
vals_tn.append(len(df[df.nuclei==3]))
vals_nn.append(len(df[df.nuclei==0]))

print("15MAR2022")
print("Number of cells:", len(df))
print("Number of cells with one nucleus:", len(df[df.nuclei == 1]))
print("Number of binucleated cells:", len(df[df.nuclei==2]))
print("----")

df = pd.read_csv("./final_results/16MAR2022_results.csv")
df = df.drop(columns=["irradiation date","cell line","radiation type","image_nr"])
df = df.astype(np.float_)
vals_bn.append(len(df[df.nuclei==2]))
vals_tot.append(len(df))
vals_sn.append(len(df[df.nuclei==1]))
vals_tn.append(len(df[df.nuclei==3]))
vals_nn.append(len(df[df.nuclei==0]))

print("16MAR2022")
print("Number of cells:", len(df))
print("Number of cells with one nucleus:", len(df[df.nuclei == 1]))
print("Number of binucleated cells:", len(df[df.nuclei==2]))
print("----")

df = pd.read_csv("./final_results/05MAY2022_results.csv")
df = df.drop(columns=["irradiation date","cell line","radiation type","image_nr"])
df = df.astype(np.float_)
vals_bn.append(len(df[df.nuclei==2]))
vals_tot.append(len(df))
vals_sn.append(len(df[df.nuclei==1]))
vals_tn.append(len(df[df.nuclei==3]))
vals_nn.append(len(df[df.nuclei==0]))

print("05MAY2022")
print("Number of cells:", len(df))
print("Number of cells with one nucleus:", len(df[df.nuclei == 1]))
print("Number of binucleated cells:", len(df[df.nuclei==2]))
print("----")

df = pd.read_csv("./final_results/03JUN2022_results.csv")
df = df.drop(columns=["irradiation date","cell line","radiation type","image_nr"])
df = df.astype(np.float_)
vals_bn.append(len(df[df.nuclei==2]))
vals_tot.append(len(df))
vals_sn.append(len(df[df.nuclei==1]))
vals_tn.append(len(df[df.nuclei==3]))
vals_nn.append(len(df[df.nuclei==0]))

print("03JUN2022")
print("Number of cells:", len(df))
print("Number of cells with one nucleus:", len(df[df.nuclei == 1]))
print("Number of binucleated cells:", len(df[df.nuclei==2]))
print("----")

df = pd.read_csv("./final_results/06SEP2022_results.csv")
df = df.drop(columns=["irradiation date","cell line","radiation type","image_nr"])
df = df.astype(np.float_)
vals_bn.append(len(df[df.nuclei==2]))
vals_tot.append(len(df))
vals_sn.append(len(df[df.nuclei==1]))
vals_tn.append(len(df[df.nuclei==3]))
vals_nn.append(len(df[df.nuclei==0]))

print("06SEP2022")
print("Number of cells:", len(df))
print("Number of cells with one nucleus:", len(df[df.nuclei == 1]))
print("Number of binucleated cells:", len(df[df.nuclei==2]))
print("----")


df = pd.read_csv("./final_results/19SEP2022_results.csv")
df = df.drop(columns=["irradiation date","cell line","radiation type","image_nr"])
df = df.astype(np.float_)
vals_bn.append(len(df[df.nuclei==2]))
vals_tot.append(len(df))
vals_sn.append(len(df[df.nuclei==1]))
vals_tn.append(len(df[df.nuclei==3]))
vals_nn.append(len(df[df.nuclei==0]))

print("19SEP2022")
print("Number of cells:", len(df))
print("Number of cells with one nucleus:", len(df[df.nuclei == 1]))
print("Number of binucleated cells:", len(df[df.nuclei==2]))
print("----")

df_ = pd.DataFrame(np.c_[vals_tot,vals_nn,vals_sn,vals_bn],
    columns=["total", "zero", "one", "two"],
    index=["15MAR2022", "16MAR2022", "05MAY2022", "03JUN2022", "06SEP2022", "19SEP2022"])
df_["other"] = df_.total - df_.zero - df_.one - df_.two
df_ = df_.drop(columns=["total"])

df_.plot(kind="bar", rot=30, figsize=(10,6), color=["navy", "blue", "dodgerblue", "skyblue"], fontsize=13)
plt.title("Cells identified by segmentation analysis", fontsize=20)
plt.xlabel("Irradiation date", fontsize=15)
plt.ylabel("Number of cells", fontsize=15)
plt.legend(title="Number of\n    nuclei", labels=["None", "One", "Two", "Other"], fontsize=13, title_fontsize=13)

"""
df_["ratio"] = (df_.one + df_.two) / (df_.zero + df_.other)
ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.scatter(ax1.get_xticks(), df_.ratio, marker="D", color="coral", s=150)
ax2.set_ylabel("(one + two) / (zero + other)", fontsize=20)
ax2.set_ylim(1,4.0)
"""
plt.tight_layout()
plt.savefig("./Figures/cells_found.png", bbox_inches="tight", dpi=200)
plt.show()
