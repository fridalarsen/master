import numpy as np
import matplotlib.pyplot as plt

import skimage.io as skio
import skimage.color as skcol

from program_v1 import pre_process, segmentation_v1, nuclei_per_cell

"""
Program showing how to use the segmentation algorithms to find cells and
nuclei in confocal images.
"""

# load images
cell_img = skio.imread("./MN_raw_data/15MAR2022/4.11gy-p1-2/9_ch00.tif")
nuclei_img = skio.imread("./MN_raw_data/15MAR2022/4.11gy-p1-2/9_ch01.tif")

# pre-process images
cell_img_processed = pre_process(cell_img, filter="Blur", size=15, equalize="Regular")
nuclei_img_processed = pre_process(nuclei_img, filter="Blur", size=3, equalize=None)

# initiate segmentation class -- one for each channel
seg_cells = segmentation_v1(cell_img_processed)
seg_nuclei = segmentation_v1(nuclei_img_processed)

# set parameters for filling mask (optional)
seg_cells.set_fill_params(n_bg=3, n_fg=3, thresh_bg=0.5, thresh_fg=0.5)
seg_nuclei.set_fill_params(n_bg=2, n_fg=3, thresh_bg=0.7, thresh_fg=0.42)

# perform segmentation (find objects)
seg_cells.find_blobs(min_dist=15, min_size=1000)
seg_nuclei.find_blobs(min_dist=8, min_size=0)

# extract found objects from class
cells = seg_cells.get_labels()
nuclei = seg_nuclei.get_labels()

# assign nuclei to cells, determine number of micronuclei in each cell
n, mn = nuclei_per_cell(cells, nuclei, mn_thresh=350)

# print results
print("Number of nuclei per cell:", n)
print("Number of micronuclei per cell:", mn)

# plot results
plt.imshow(skcol.label2rgb(nuclei, bg_label=0))
plt.imshow(skcol.label2rgb(cells, bg_label=0), alpha=0.4)
plt.axis("off")

plt.title("Found cells and nuclei", fontsize=20)

plt.show()
