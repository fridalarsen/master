import numpy as np
import matplotlib.pyplot as plt

import skimage.io as skio
import skimage.color as skcol

from program_v1 import pre_process, segmentation_v1, nuclei_per_cell

"""
General program for showing 8 images from an experiment, used to determine the
 best parameters to be used for analysis.
"""

rad_date = "03JUN2022"

fig, axs = plt.subplots(2, 4, figsize=(9,7))

# specify index of images
a = [1,2,3,4,5,6,7,8]

for i in range(len(a)):
    cell_img = skio.imread("./MN_raw_data/{}/0gy-1/{}_ch00.tif".format(rad_date, a[i]))
    nuc_img = skio.imread("./MN_raw_data/{}/0gy-1/{}_ch01.tif".format(rad_date, a[i]))

    # specify parameters given to pre_process and segmentation_v1
    # pre-process images
    cell_img = pre_process(cell_img, filter="Blur", size=5, equalize="Regular")
    nuc_img = pre_process(nuc_img, filter="Blur", equalize=None, size=5)

    # find cells
    v1_cells = segmentation_v1(cell_img)
    v1_cells.set_fill_params(n_bg=3, n_fg=3, thresh_bg=0.5, thresh_fg=0.45)
    v1_cells.find_blobs(min_dist=15, min_size=1000)
    cells = v1_cells.get_labels()

    # find nuclei
    v1_nuclei = segmentation_v1(nuc_img)
    v1_nuclei.set_fill_params(n_bg=3, n_fg=3, thresh_bg=0.7, thresh_fg=0.42)
    v1_nuclei.find_blobs(min_dist=9, min_size=0)
    nuclei = v1_nuclei.get_labels()

    if i in [0,1,2,3]:
        axs[0,i].imshow(skcol.label2rgb(nuclei, bg_label=0), alpha=1)
        axs[0,i].imshow(skcol.label2rgb(cells, bg_label=0), alpha=0.4)
        axs[0,i].axis("off")
    if i in [4,5,6,7]:
        axs[1,i-4].imshow(skcol.label2rgb(nuclei, bg_label=0), alpha=1)
        axs[1,i-4].imshow(skcol.label2rgb(cells, bg_label=0), alpha=0.4)
        axs[1,i-4].axis("off")

fig.suptitle("Found cells and nuclei", fontsize=20)
fig.tight_layout()
plt.show()
