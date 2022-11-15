import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std

from survival_curve import survival_fraction

# load experimental data and plot survival curve
dates = ["02SEP2021", "08SEP2021", "03NOV2021", "10NOV2021", "09DES2021",
         "26JAN2022", "10FEB2022"]

multiplicities = []
doses = []
survival = []

colors = ["hotpink", "chartreuse", "forestgreen", "orangered", "darkviolet",
          "darkblue", "steelblue"]
for date in dates:
    dose, SF, MP = survival_fraction(date=date, MP_correction=True)

    multiplicities.append(MP)
    survival.append(SF)

    for d in dose:
        if d not in doses:
            doses.append(d)

    plt.plot(dose, SF, c=colors[dates.index(date)], ls="--")
    plt.scatter(dose, SF, s=30, c=colors[dates.index(date)], label=date)
    plt.yscale("log")
plt.xlabel("Dose [Gy]", fontsize=15)
plt.ylabel("Survival", fontsize=15)
plt.title("Survival data for MOC2", fontsize=20)
plt.legend(prop={"size":12}, loc=3)
plt.savefig("./Figures/moc2_all_experiments.png", bbox_inches="tight", dpi=200)
plt.show()

# excluding 02SEP2021 and 10FEB2022 (outliers) from average
survival = survival[1:-1]
dates = dates[1:-1]

# compute averages and std
avg_SF = []
std_SF = []
for i in range(len(doses)):
    SFs = []
    for j in range(len(dates)):
        if i < len(survival[j]):
            SFs.append(survival[j][i])

    avg_SF.append(np.mean(SFs))
    std_SF.append(np.std(SFs))

    SFs = []

plt.errorbar(doses, avg_SF, yerr=std_SF, capsize=4, marker="D", color="deepskyblue")
plt.yscale("log")
plt.xlabel("Dose [Gy]", fontsize=15)
plt.ylabel("Survival", fontsize=15)
plt.title("Average survival of MOC2-cells", fontsize=20)
plt.savefig("./Figures/moc2_survival_curve.png", bbox_inches="tight", dpi=200)
plt.show()

# find LQ-parameters
def S(D, alpha, beta):
    """
    LQ-model.
    Arguments:
        D (float or array): Dose or doses.
        alpha (float): Linear dose contribution.
        beta (float): Squared dose contribution.
    Returns:
        S (float or array): Survival fraction. Same type as D.
    """
    S = np.exp(-alpha*D-beta*(D**2))

    return S

# make design matrix
pf = PolynomialFeatures(degree=2, include_bias=False)
xp = pf.fit_transform(np.array(doses).reshape(-1,1))

# log of S, since we want to use linear regression
y = np.log(avg_SF)

# ordinary least squares
model = sm.OLS(y,-xp)
results = model.fit()

# print parameters and confidence interval
confidence_intervals = results.conf_int(alpha=0.01)

print("Alpha:", results.params[0])
print("Confidence interval alpha:", confidence_intervals[0])

print("Beta", results.params[1])
print("Confidence interval beta:", confidence_intervals[1])

D = np.linspace(0,10,100)

# find confidence interval of prediction
xpred = pf.fit_transform(D.reshape(-1,1))
ypred = results.predict(-xpred)

prediction_std, upper, lower = wls_prediction_std(results, exog=-xpred, alpha = 0.01)

plt.fill_between(D, np.exp(upper), np.exp(lower), alpha=0.4, label="99% confidence interval", color="skyblue")
plt.scatter(doses, avg_SF, marker="D", color="navy", label="Average survival fraction")
plt.plot(D, S(D, results.params[0], results.params[1]), color="dodgerblue", label="LQ-model")
plt.legend(prop={"size":12}, loc=3)
plt.yscale("log")
plt.xlabel("Dose [Gy]", fontsize=15)
plt.ylabel("Survival", fontsize=15)
plt.title("LQ-model for MOC2 cells", fontsize=20)
plt.savefig("./Figures/MOC2_LQ_model.png", bbox_inches="tight", dpi=200)
plt.show()

# comparison of LQ-model for MOC1, MOC2 and A549
alpha_a549 = 0.400
beta_a549 = 0.031

alpha_moc1 = 0.250
beta_moc1 = 0.039

plt.plot(D, S(D, results.params[0], results.params[1]), label="MOC2", color="deepskyblue")
plt.plot(D, S(D, alpha_moc1, beta_moc1), label="MOC1", color="royalblue")
plt.plot(D, S(D, alpha_a549, beta_a549), label="A549", color="lightseagreen")
plt.legend(prop={"size":12}, loc=3)
plt.yscale("log")
plt.xlabel("Dose [Gy]", fontsize=15)
plt.ylabel("Survival", fontsize=15)
plt.title("LQ-model for different cell lines", fontsize=20)
plt.savefig("./Figures/LQ_comparison.png", bbox_inches="tight", dpi=200)
plt.show()


"""
sample run:

$ python3 make_survival_curve.py
Alpha: 0.1692881494477032
Confidence interval alpha: [0.09645815 0.24211815]
Beta 0.0415619517237503
Confidence interval beta: [0.03310286 0.05002104]

"""
