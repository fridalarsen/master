import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fsolve

def read_data(date, data_folder):
    """
    Function for reading survival curve data given the date of the experiment.
    Arguments:
        date (str): Date of experiment. Alternatively, filename containing
                    experiment data.
        data_folder (str): Folder containing data.
    Returns:
        dose (array): Doses used in experiment.
        counts (array): Counts of flasks for each dose. In format [[counts dose 1], [counts dose 2], ...]
        ctr (array): Counts of control flasks (dose = 0 Gy).
    Raises:
        TypeError: If date is not a string.
    """
    if not isinstance(date, str):
        raise TypeError("The date must be given as a string.")
    else:
        filename = f"./{data_folder}/{date}.csv"

    try:
        file = open(filename)
    except FileNotFoundError:
        print("Could not find a datafile for this date. Check urself.")

    lines = file.read().split()

    dose = []
    count = []
    count_temp = []
    for i in range(1, len(lines)):
        line = lines[i].split(";")

        d = float(line[0])
        c = float(line[1])

        if len(dose) == 0:
            dose.append(d)
            count_temp.append(c)
        elif d not in dose and len(dose) != 0:
            count.append(count_temp)
            count_temp = []

            dose.append(d)
            count_temp.append(c)
        else:
            count_temp.append(c)

    count.append(count_temp)

    return dose, count

def survival_fraction(date, data_folder="data_MOC2", MP_correction=True):
    """
    Function for finding the surviving fraction of a clonogenic assay
    experiment.
    Requires an experiment file, a CFU-file and a seedded file saved within a
    data folder.
    Arguments:
        date (str): Date of experiment.
        data_folder (str, optional): Folder containing the data sets.
                                     Defaults to "data_MOC2".
        MP_correction (bool, optional): Whether to include a multiplicity
                                        correction.
                                        Defaults to True.
    Returns:
        doses (array): Doses used in experiment.
        SF (array): Survival fraction for each dose.
        MP (float): Multiplicity of experiment.
    """
    doses, counts = read_data(date, data_folder)

    # find average counts for each set of flasks
    avg_counts = [np.mean(counts[i]) for i in range(len(doses))]

    # find number of cells seeded in each set of flasks
    file = open(f"./{data_folder}/seeded.csv")
    lines = file.read().split()

    idx = lines[0].split(";").index(date)

    seeded = []
    for i in range(1, len(doses)+1):
        line = lines[i].split(";")
        seeded.append(float(line[idx]))

    file.close()

    # determine plating efficiency
    PE = avg_counts[0]/seeded[0]

    # determine naive survival fraction
    SF_obs = [avg_counts[i]/(seeded[i]*PE) for i in range(1, len(doses))]

    # find CFU info for multiplicity correction
    file = open(f"./{data_folder}/cfu_counts.csv")
    lines = file.read().split()

    idx = lines[0].split(";").index(date)

    CFU_n = []
    CFU_counts = []
    for i in range(1, len(lines)):
        line = lines[i].split(";")
        CFU_n.append(line[0])
        CFU_counts.append(float(line[idx]))

    file.close()

    U = [x/sum(CFU_counts) for x in CFU_counts]

    MP = sum([U[i]*(i+1) for i in range(len(U))])

    # perform multiplicity correction
    if MP_correction:
        # determine survival fraction with multiplicity correction
        def f(S, U, SF):
            sum = 0
            for i in range(len(U)):
                sum += U[i]*(1-((1-S)**(i+1)))

            return sum - SF

        SF_corrected = []
        for SF in SF_obs:
            corrected = fsolve(func=f, x0=[SF], args=(U, SF))
            SF_corrected.append(corrected[0])

        # insert survival of control for plotting purposes
        SF_corrected.insert(0, 1.0)

        return doses, SF_corrected, MP


    else:
        # insert survival of control for plotting purposes
        SF_obs.insert(0, 1.0)

        return doses, SF_obs, MP


if __name__ == "__main__":
    date = "02SEP2021"

    doses, SF, MP = survival_fraction(date=date, MP_correction=True)

    print(doses)
    print(SF)

    plt.scatter(doses, SF, label=f"MP={MP:.2f}")
    plt.xlabel("Dose [Gy]", fontsize=12)
    plt.ylabel("Surviving fraction", fontsize=12)
    plt.title("Survival curve 02SEP2021", fontsize=15)
    plt.legend()
    plt.show()
