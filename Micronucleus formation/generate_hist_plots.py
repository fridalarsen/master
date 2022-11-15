import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib.container import BarContainer

scaler = MinMaxScaler()

def generate_pivot_table(results_file, only_binucliated=True, rad_type="proton"):
    # read files
    df = pd.read_csv(results_file)
    df = df.drop(columns=["irradiation date","cell line","radiation type","image_nr"])
    df = df.astype(np.float_)
    if rad_type == "proton":
        df.position[pd.isna(df.position)] = 0

    # remove non-binucliated cells
    if only_binucliated:
        df = df[df.nuclei == 2]
    df = df.drop(columns=["nuclei"])

    if rad_type == "proton":
        # group data
        df_grouped = df.groupby(["dose","position","sample_id"]).micronuclei.value_counts()
        df_grouped = df_grouped.rename("counts").astype(np.int_).reset_index()

        # prepare pivot table
        pivot_table = df_grouped.pivot_table(index=["dose","position","sample_id"], columns=["micronuclei"])
        pivot_table = pivot_table.fillna(0).astype(np.int_)

        # remove multilevel column-index
        pivot_table.columns = pivot_table.columns.droplevel(0)
        pivot_table = pivot_table.rename_axis(None, axis=1)
    elif rad_type == "x-ray":
        # group data
        df_grouped = df.groupby(["dose", "sample_id"]).micronuclei.value_counts()
        df_grouped = df_grouped.rename("counts").astype(np.int_).reset_index()

        # prepare pivot table
        pivot_table = df_grouped.pivot_table(index=["dose","sample_id"], columns=["micronuclei"])
        pivot_table = pivot_table.fillna(0).astype(np.int_)

        # remove multilevel column-index
        pivot_table.columns = pivot_table.columns.droplevel(0)
        pivot_table = pivot_table.rename_axis(None, axis=1)
    else:
        raise ValueError("rad_type must be proton or x-ray.")

    return pivot_table

def normalize_pivot_table(pt, scale_percent=True):
    # copy pivot table and normalize row-wise
    pt_norm = pt.copy()
    pt_norm["total"] = pt_norm.sum(axis=1)
    pt_norm = pt_norm.iloc[:,:-1].truediv(pt_norm.total, axis=0)

    # scale to 100%
    if scale_percent:
        pt_norm = 100 * pt_norm

    return pt_norm


# 15MAR2022
pt = generate_pivot_table("./final_results/15MAR2022_results.csv", rad_type="proton")
pt_norm = normalize_pivot_table(pt)

# stacked count bars
pt.plot(kind="bar", stacked=True, rot=0, figsize=(10,6))

labels = ["Ctr 1", "Ctr 2"] + [f"{d} Gy\np{p:n}" for d,p,_ in pt.index][2:]
plt.xticks(ticks=plt.gca().get_xticks(), labels=labels, fontsize=15)
plt.xlabel("")
plt.ylabel("Number of micronuclei", fontsize=20)
plt.legend(title="#MN", labels=range(pt.columns.size))
plt.title("A549 - protons\n15MAR2022", fontsize=20)
plt.show()

# normalized stacked count bars
pt_norm.plot(kind="bar", stacked=True, rot=0, figsize=(10,6))

plt.xticks(ticks=plt.gca().get_xticks(), labels=labels, fontsize=15)
plt.xlabel("")
plt.ylabel("Distribution of #micronuclei [%]", fontsize=20)
plt.legend(title="#MN", labels=range(pt.columns.size), bbox_to_anchor=(1, 0.75))
plt.title("A549 - protons\n15MAR2022", fontsize=20)
plt.show()

# 16MAR2022
pt = generate_pivot_table("./final_results/16MAR2022_results.csv", rad_type="proton")
pt_norm = normalize_pivot_table(pt)

# stacked count bars
pt.plot(kind="bar", stacked=True, rot=0, figsize=(10,6))

labels = ["Ctr 1", "Ctr 2"] + [f"{d} Gy\np{p:n}" for d,p,_ in pt.index][2:]
plt.xticks(ticks=plt.gca().get_xticks(), labels=labels, fontsize=15)
plt.xlabel("")
plt.ylabel("Number of micronuclei", fontsize=20)
plt.legend(title="#MN", labels=range(pt.columns.size))
plt.title("A549 - protons\n16MAR2022", fontsize=20)
plt.show()

# normalized stacked count bars
pt_norm.plot(kind="bar", stacked=True, rot=0, figsize=(10,6))

plt.xticks(ticks=plt.gca().get_xticks(), labels=labels, fontsize=15)
plt.xlabel("")
plt.ylabel("Distribution of #micronuclei [%]", fontsize=20)
plt.legend(title="#MN", labels=range(pt.columns.size), bbox_to_anchor=(1, 0.75))
plt.title("A549 - protons\n16MAR2022", fontsize=20)
plt.show()



# 05MAY2022
pt = generate_pivot_table("./final_results/05MAY2022_results.csv", rad_type="proton")
pt_norm = normalize_pivot_table(pt)
pt_norm = pt_norm.reorder_levels(["position","dose","sample_id"]).sort_index()
labels = ["Ctr 1", "Ctr 2"] + [f"{d} Gy p{p:n}" for p,d,_ in pt_norm.index][2:]


# stacked count bars
pt.plot(kind="bar", stacked=True, rot=0, figsize=(10,6))

plt.xticks(ticks=plt.gca().get_xticks(), labels=labels, fontsize=15)
plt.xlabel("")
plt.ylabel("Number of micronuclei", fontsize=20)
plt.legend(title="#MN", labels=range(pt.columns.size))
plt.title("MOC2 - protons\n05MAY2022", fontsize=20)
plt.show()

# normalized stacked count bars
x = np.linspace(0,13,14)
ax = pt_norm.plot.barh(stacked=True, rot=0, figsize=(9,8), colormap="jet")
y_coords = np.unique([bar.get_xy()[1] for bar in ax.patches])
y_new = np.zeros(y_coords.shape)
y_new[ :2] = 0.75*np.linspace(0,1,2)
y_new[2:8] = y_new[1] + 1.5 + 0.75*np.linspace(0,5,6)
y_new[8: ] = y_new[7] + 1.5 + 0.75*np.linspace(0,5,6)

for i,y in enumerate(y_coords):
    for bar in ax.patches:
        if bar.get_xy()[1] == y:
            bar.set_y(y_new[i])

X = pt_norm.values.copy()
for i,N in enumerate(pt_norm.columns):
    if i > 0:
        X[:,i] = X[:,i-1] + pt_norm[N]

x_means = X.mean(axis=1)
#plt.scatter(x_means, y_new+0.25, marker="^", s=50, color="red")


plt.text(-30, -0.25 +  y_new[:1].mean(), "Control", fontsize=20, rotation=90)
plt.text(-30, -0.75 +  y_new[2:8].mean(), "Position 1", fontsize=20, rotation=90)
plt.text(-30, -0.75 +  y_new[8:].mean(), "Position 5", fontsize=20, rotation=90)

labels = ["1", "2"] + [f"{d:.2f} Gy" for p,d,_ in pt_norm.index][2:]

plt.ylim(-0.5,12.25)
plt.yticks(ticks=y_new+0.25, labels=labels, fontsize=15)
plt.ylabel("")
plt.xlabel("Number of cells [%]", fontsize=20)
plt.legend(title="Number of\nmicronuclei", labels=range(pt.columns.size), bbox_to_anchor=(1.02, 0.75), fontsize=13, title_fontsize=15)
plt.title("Distribution of micronuclei in proton\nirradiated MOC2-cells (05MAY2022)", fontsize=20)
plt.xlim(0,100)
plt.subplots_adjust(left=0)
plt.tight_layout()
plt.savefig("./Figures/mn_dist_05MAY2022.png", bbox_inches="tight", dpi=200)
plt.show()



# 06SEP2022
pt = generate_pivot_table("./final_results/06SEP2022_results.csv", rad_type="proton")
pt_norm = normalize_pivot_table(pt)

# stacked count bars
pt.plot(kind="bar", stacked=True, rot=0, figsize=(10,6))

labels = ["Ctr 1", "Ctr 2"] + [f"{d} Gy\np{p:n}" for d,p,_ in pt.index][2:]
plt.xticks(ticks=plt.gca().get_xticks(), labels=labels, fontsize=15)
plt.xlabel("")
plt.ylabel("Number of micronuclei", fontsize=20)
plt.legend(title="#MN", labels=range(pt.columns.size))
plt.title("MOC2 - protons\n06SEP2022", fontsize=20)
plt.show()

# normalized stacked count bars
pt_norm.plot(kind="bar", stacked=True, rot=0, figsize=(10,6))

plt.xticks(ticks=plt.gca().get_xticks(), labels=labels, fontsize=15)
plt.xlabel("")
plt.ylabel("Distribution of #micronuclei [%]", fontsize=20)
plt.legend(title="#MN", labels=range(pt.columns.size), bbox_to_anchor=(1, 0.75))
plt.title("MOC2 - protons\n06SEP2022", fontsize=20)
plt.show()

# 03JUN2022
pt = generate_pivot_table("./final_results/03JUN2022_results.csv", rad_type="x-ray")
pt_norm = normalize_pivot_table(pt)

# stacked count bars
pt.plot(kind="bar", stacked=True, rot=0, figsize=(10,6))

labels = ["Ctr 1"] + [f"{d} Gy" for d,_ in pt.index][1:]
plt.xticks(ticks=plt.gca().get_xticks(), labels=labels, fontsize=15)
plt.xlabel("")
plt.ylabel("Number of micronuclei", fontsize=20)
plt.legend(title="#MN", labels=range(pt.columns.size))
plt.title("A549 - x-rays\n03JUN2022", fontsize=20)
plt.show()

# normalized stacked count bars
pt_norm.plot(kind="bar", stacked=True, rot=0, figsize=(10,6))

plt.xticks(ticks=plt.gca().get_xticks(), labels=labels, fontsize=15)
plt.xlabel("")
plt.ylabel("Distribution of #micronuclei [%]", fontsize=20)
plt.legend(title="#MN", labels=range(pt.columns.size), bbox_to_anchor=(1, 0.75))
plt.title("A549 - x-rays\n03JUN2022", fontsize=20)
plt.show()

# 19SEP2022
pt = generate_pivot_table("./final_results/19SEP2022_results.csv", rad_type="x-ray")
pt_norm = normalize_pivot_table(pt)

# stacked count bars
pt.plot(kind="bar", stacked=True, rot=0, figsize=(10,6))

labels = ["Ctr 1", "Ctr 2"] + [f"{d} Gy" for d,_ in pt.index][2:]
plt.xticks(ticks=plt.gca().get_xticks(), labels=labels, fontsize=15)
plt.xlabel("")
plt.ylabel("Number of micronuclei", fontsize=20)
plt.legend(title="#MN", labels=range(pt.columns.size))
plt.title("MOC2 - x-rays\n19SEP2022", fontsize=20)
plt.show()

# normalized stacked count bars
pt_norm.plot(kind="bar", stacked=True, rot=0, figsize=(10,6))

plt.xticks(ticks=plt.gca().get_xticks(), labels=labels, fontsize=15)
plt.xlabel("")
plt.ylabel("Distribution of #micronuclei [%]", fontsize=20)
plt.legend(title="#MN", labels=range(pt.columns.size), bbox_to_anchor=(1, 0.75))
plt.title("MOC2 - x-rays\n19SEP2022", fontsize=20)

plt.show()


# make csv file for number of nuclei in cells
dates = ["15MAR2022","16MAR2022","05MAY2022","03JUN2022","06SEP2022","19SEP2022"]
cols = ["irr. date","position","dose","sampleID","total","1","2","2/1"]
rads = ["proton","proton","proton","x-ray","proton","x-ray"]
DF = []
for date,rad in zip(dates,rads):
    df = generate_pivot_table(f"./final_results/{date}_results.csv",
        rad_type=rad, only_binucliated=False)
    df["total"] = df.sum(axis=1)
    df["2/1"] = df[2]/ df[1]
    df["irradiation-date"] = date
    df = df.reset_index()
    if "position" not in df.columns:
        df["position"] = np.nan
    df = df[["irradiation-date","position","dose","sample_id","total",1,2,"2/1"]]
    DF.append(df.rename(columns={a:b for a,b in zip(df.columns,cols)}))

DF = pd.concat(DF)
DF = DF.astype({
    "irr. date" : str,
    "position" : "Int64",
    "sampleID" : np.int_,
    "total" : np.int_,
    "1" : np.int_,
    "2" : np.int_
})
DF.to_csv("./final_results/nuclei_summary.csv", na_rep="NA", index=False, float_format="%.2f")
