import numpy as np
import matplotlib.pyplot as plt

import skimage.io as skio
import skimage.color as skcol
import skimage.filters as skfilt
from scipy import ndimage

from program_v1 import pre_process, segmentation_v1

"""
Program for creating illustrations of the image analyzing process for thesis.
"""

# load example images
cell_img = skio.imread("./MN_raw_data/15MAR2022/4.11gy-p1-2/9_ch00.tif")
nuclei_img = skio.imread("./MN_raw_data/15MAR2022/4.11gy-p1-2/9_ch01.tif")

"""
# plot images
figure, axes = plt.subplots(1,2)

axes[0].imshow(cell_img, cmap="Greens_r")
axes[0].set_title("Cells", fontsize=15)
axes[0].axis("off")

axes[1].imshow(nuclei_img, cmap="Reds_r")
axes[1].set_title("Nuclei", fontsize=15)
axes[1].axis("off")

figure.suptitle("Confocal images\nof MOC2 cells", fontsize=20)
plt.tight_layout()
#plt.savefig("./Figures/confocal_images.png", bbox_inches="tight", dpi=200)
plt.show()
"""

# pre-process images
cell_img_processed = pre_process(cell_img, filter="Blur", size=15, equalize="Regular")
nuclei_img_processed = pre_process(nuclei_img, filter="Blur", size=3, equalize=None)

# initiate segmentation class
seg_cells = segmentation_v1(cell_img_processed)
seg_nuclei = segmentation_v1(nuclei_img_processed)

# set parameters for filling mask
seg_cells.set_fill_params(n_bg=3, n_fg=3, thresh_bg=0.5, thresh_fg=0.5)
seg_nuclei.set_fill_params(n_bg=2, n_fg=3, thresh_bg=0.7, thresh_fg=0.42)

# perform segmentation (find objects)
seg_cells.find_blobs(min_dist=15, min_size=1000)
seg_nuclei.find_blobs(min_dist=8, min_size=0)

# extract images of various stages in segmentation
cells_otsu_mask = seg_cells.get_mask()
nuclei_otsu_mask = seg_nuclei.get_mask()

cells_EDT = seg_cells.get_EDT()
nuclei_EDT = seg_nuclei.get_EDT()

cells_markers = seg_cells.get_markers()
nuclei_markers = seg_nuclei.get_markers()

cells = seg_cells.get_labels()
nuclei = seg_nuclei.get_labels()

# make figure
figure = plt.figure(constrained_layout=True, figsize=(6,6))
subfigs = figure.subfigures(2, 1)

original_images = subfigs[0].subplots(1,2)
original_images[0].imshow(cell_img, cmap="Greens_r")
original_images[0].set_title("Cell channel", fontsize=15)
original_images[0].axis("off")

original_images[1].imshow(nuclei_img, cmap="Reds_r")
original_images[1].set_title("Nucleus channel", fontsize=15)
original_images[1].axis("off")

subfigs[0].suptitle("Original images", fontsize=18)

pre_process = subfigs[1].subplots(1,2)
pre_process[0].imshow(cell_img_processed, cmap="Greens_r")
pre_process[0].set_title(" ", fontsize=15)
pre_process[0].axis("off")

pre_process[1].imshow(nuclei_img_processed, cmap="Reds_r")
pre_process[1].set_title(" ", fontsize=15)
pre_process[1].axis("off")

subfigs[1].suptitle("Pre-processed images", fontsize=18)

plt.savefig("./Figures/method_summary2.png", bbox_inches="tight", dpi=200)
plt.show()


figure = plt.figure(constrained_layout=True, figsize=(6,10))
subfigs = figure.subfigures(4, 1)

otsu = subfigs[0].subplots(1,2)
otsu[0].imshow(cells_otsu_mask, cmap="Purples")
otsu[0].axis("off")
otsu[0].set_title("Cell channel", fontsize=15)

otsu[1].imshow(nuclei_otsu_mask, cmap="Purples")
otsu[1].axis("off")
otsu[1].set_title("Nucleus channel", fontsize=15)

subfigs[0].suptitle("Separation of foreground and background\nby Otsu thresholding", fontsize=18)

edt = subfigs[1].subplots(1,2)
edt[0].imshow(cells_EDT, cmap="Blues")
edt[0].axis("off")

edt[1].imshow(nuclei_EDT, cmap="Blues")
edt[1].axis("off")

subfigs[1].suptitle("Euclidean distance transform computed\nfor all foreground pixels", fontsize=18)

seg = subfigs[2].subplots(1,2)
seg[0].imshow(skcol.label2rgb(cells, bg_label=0))
seg[0].axis("off")

seg[1].imshow(skcol.label2rgb(nuclei, bg_label=0))
seg[1].axis("off")

subfigs[2].suptitle("Watershed segmentation for splitting\nforeground into objects", fontsize=18)

assign = subfigs[3].subplots(1,1)
assign.imshow(skcol.label2rgb(nuclei, bg_label=0))
assign.imshow(skcol.label2rgb(cells, bg_label=0), alpha=0.4)
assign.axis("off")

subfigs[3].suptitle("Assign nuclei to cells", fontsize=18)

#figure.suptitle("Image segmentation", fontsize=20)

plt.savefig("./Figures/method_summary1.png", bbox_inches="tight", dpi=200)
plt.show()

"""
figure, axes = plt.subplots(5, 2, figsize=(10,8))

# original images
axes[0,0].imshow(cell_img, cmap="Greens_r")
axes[0,0].set_title("Orignal image - cells", fontsize=15)
axes[0,0].axis("off")

axes[0,1].imshow(nuclei_img, cmap="Reds_r")
axes[0,1].set_title("Original image - nuclei", fontsize=15)
axes[0,1].axis("off")

# pre-processed images
axes[1,0].imshow(cell_img_processed, cmap="Greens_r")
axes[1,0].set_title("Filtered image - cells", fontsize=15)
axes[1,0].axis("off")

axes[1,1].imshow(nuclei_img_processed, cmap="Reds_r")
axes[1,1].set_title("Filtered image - nuclei", fontsize=15)
axes[1,1].axis("off")

# Otsu masks
axes[2,0].imshow(cells_otsu_mask, cmap="Purples")
axes[2,0].set_title("Otsu mask - cells", fontsize=15)
axes[2,0].axis("off")

axes[2,1].imshow(nuclei_otsu_mask, cmap="Purples")
axes[2,1].set_title("Otsu mask - nuclei", fontsize=15)
axes[2,1].axis("off")

# EDT
axes[3,0].imshow(cells_EDT, cmap="Purples")
axes[3,0].set_title("EDT - cells", fontsize=15)
axes[3,0].axis("off")

axes[3,1].imshow(nuclei_EDT, cmap="Purples")
axes[3,1].set_title("EDT - nuclei", fontsize=15)
axes[3,1].axis("off")

# found objects
cells_ = skcol.label2rgb(cells, bg_label=0)
axes[4,0].imshow(cells_)
axes[4,0].set_title("Found cells", fontsize=15)
axes[4,0].axis("off")

nuclei_ = skcol.label2rgb(nuclei, bg_label=0)
axes[4,1].imshow(nuclei_)
axes[4,1].set_title("Found nuclei", fontsize=15)
axes[4,1].axis("off")

plt.tight_layout()

plt.savefig("./Figures/program_flow.png", dpi=200)
plt.show()

"""
"""

# pre-processing illustration
nuclei_img = skio.imread("./MN_raw_data/05MAY2022/8.04gy-p1-1/4_ch01.tif")
nuclei_blurry = pre_process(nuclei_img, filter="Blur", size=15, equalize=None)

figure, axes = plt.subplots(1,2)

axes[0].imshow(nuclei_img, cmap="Reds_r")
axes[0].set_title("Original image", fontsize=15)
axes[0].axis("off")

axes[1].imshow(nuclei_blurry, cmap="Reds_r")
axes[1].set_title("Blurred image", fontsize=15)
axes[1].axis("off")

#plt.savefig("./Figures/blurred_nuclei.png", bbox_inches='tight', dpi=200)
plt.show()


# pre-processing (both cells and nuclei)
cell_img = skio.imread("./MN_raw_data/05MAY2022/8.04gy-p1-1/4_ch00.tif")
nuclei_img = skio.imread("./MN_raw_data/05MAY2022/8.04gy-p1-1/4_ch01.tif")

cell_img_processed = pre_process(cell_img, filter="Blur", size=19, equalize="Regular")
nuclei_img_processed = pre_process(nuclei_img, filter="Blur", size=3, equalize=None)

figure = plt.figure(constrained_layout=True, figsize=(4,4))
subfigs = figure.subfigures(2, 1, wspace=0.1)

top = subfigs[0].subplots(1,2)
top[0].imshow(cell_img, cmap="Greens_r")
top[0].axis("off")
top[1].imshow(nuclei_img, cmap="Reds_r")
top[1].axis("off")
subfigs[0].suptitle("Original images", fontsize=15)


bottom = subfigs[1].subplots(1,2)
bottom[0].imshow(cell_img_processed, cmap="Greens_r")
bottom[0].axis("off")
bottom[1].imshow(nuclei_img_processed, cmap="Reds_r")
bottom[1].axis("off")
subfigs[1].suptitle("Filtered images", fontsize=15)


#plt.savefig("./Figures/filtered_images.png", bbox_inches='tight', dpi=200)
plt.show()

# separation of foreground and background
thresh_og = skfilt.threshold_otsu(cell_img)
mask_og = cell_img >= thresh_og

thresh = skfilt.threshold_otsu(cell_img_processed)
mask = cell_img_processed >= thresh

figure = plt.figure(constrained_layout=True, figsize=(4.5,4.5))
subfigs = figure.subfigures(2, 1, wspace=0.1)

top = subfigs[0].subplots(1,2)
top[0].imshow(cell_img, cmap="Greens_r")
top[0].axis("off")
top[1].imshow(mask_og, cmap="Purples")
top[1].axis("off")
subfigs[0].suptitle("Otsu thresholding on\noriginal image", fontsize=15)


bottom = subfigs[1].subplots(1,2)
bottom[0].imshow(cell_img_processed, cmap="Greens_r")
bottom[0].axis("off")
bottom[1].imshow(mask, cmap="Purples")
bottom[1].axis("off")
subfigs[1].suptitle("Otsu thresholding on\nfiltered image", fontsize=15)

#plt.savefig("./Figures/otsu_mask.png", bbox_inches='tight', dpi=200)
plt.show()

# EDT
EDT = ndimage.distance_transform_edt(mask)

plt.imshow(EDT, cmap="Blues")
plt.imshow(mask, cmap="gray_r", alpha=0.2)
plt.title("Euclidean distance transform", fontsize=15)
plt.axis("off")

#plt.savefig("./Figures/edt.png", bbox_inches='tight', dpi=200)
plt.show()


# Illustration of high confluency cell images.

image = skio.imread("./MN_raw_data/18MAR2022/0gy-pNA-2/3_ch00.tif")
image_ = pre_process(image, filter="Blur", size=25, equalize="Regular")

segmentation = segmentation_v1(image_)
segmentation.set_fill_params(n_bg=3, n_fg=3, thresh_bg=0.5, thresh_fg=0.5)
segmentation.find_blobs(min_dist=15, min_size=1000)


figure = plt.figure(constrained_layout=True, figsize=(5,5))
subfigs = figure.subfigures(2, 1, wspace=0.1)

top = subfigs[0].subplots(1,2)
top[0].imshow(image, cmap="Greens_r")
top[0].axis("off")
top[0].set_title("Original image", fontsize=20)

top[1].imshow(segmentation.get_mask(), cmap="Purples")
top[1].axis("off")
top[1].set_title("Otsu mask", fontsize=20)

bottom = subfigs[1].subplots(1,1)
cells = skcol.label2rgb(segmentation.get_labels(), bg_label=0)
bottom.imshow(cells)
bottom.axis("off")
subfigs[1].suptitle("Found cells", fontsize=20)

plt.savefig("./Figures/high_confluency.png", bbox_inches="tight", dpi=200)
plt.show()
"""
