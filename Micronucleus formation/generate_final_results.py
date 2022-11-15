import numpy as np
import matplotlib.pyplot as plt

import skimage.io as skio
import skimage.color as skcol

from generate_results import generate_results

"""
General program for generating the results file for an experiment.
"""

rad_date = "19SEP2022"                      # date of irradiation
cell_line = "MOC2"                          # cell line used for experiment
rad_type = "xray"                         # type of irradiation

# parameters determined from control images
fill_param_cells = [3, 3, 0.5, 0.5]       # n_bg, n_fg, thresh_bg, thresh_fg
fill_param_nuclei = [3, 3, 0.5, 0.5]      # n_bg, n_fg, thresh_bg, thresh_fg
blob_param_cells = [15, 1000]             # min_dist, min_size
blob_param_nuclei = [9, 0]                # min_dist, min_size
filters = ["Blur", "Blur"]
filter_params = [15, 3]
equalize = ["Regular", None]
thresh_mn = 350
thresh = ["Multi", "Otsu"]

filename = f"./final_results/{rad_date}_results"

generate_results(rad_date=rad_date, cell_line=cell_line, rad_type=rad_type,
                 data_folder="MN_raw_data",
                 fill_param_cells=fill_param_cells,
                 fill_param_nuclei=fill_param_nuclei,
                 blob_param_cells=blob_param_cells,
                 blob_param_nuclei=blob_param_nuclei, equalize=equalize,
                 filters=filters, filter_params=filter_params,
                 filename=filename, mn_thresh=thresh_mn, thresh=thresh)
